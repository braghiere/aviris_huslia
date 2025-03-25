using Distributed
using ProgressMeter: @showprogress
using Emerald
using Emerald.EmeraldData.GlobalDatasets: LandDatasetLabels, grid_dict, grid_spac
using Emerald.EmeraldData.WeatherDrivers: grid_weather_driver
using Emerald.EmeraldLand.Namespace: SPACConfiguration
using Emerald.EmeraldFrontier: simulation!, SAVING_DICT
using Dates
using DataFrames
using NCDatasets
using NetcdfIO: read_nc

# Initialize Logging
logfile = open("detailed_simulation.log", "w")
log(msg) = (println(msg); println(logfile, msg))

# Initialize parallel workers
if nworkers() == 1
    addprocs(128; exeflags="--project")
end

@everywhere begin
    using Emerald
    using Emerald.EmeraldData.GlobalDatasets: LandDatasetLabels, grid_dict, grid_spac
    using Emerald.EmeraldData.WeatherDrivers: grid_weather_driver
    using Emerald.EmeraldLand.Namespace: SPACConfiguration
    using Emerald.EmeraldFrontier: simulation!, SAVING_DICT
    using Dates
    using DataFrames

    FT = Float64
    CONFIG = SPACConfiguration(FT)
end

# Setup SAVING_DICT outputs
for var in ["GPP", "CNPP", "MOD_SWC", "MOD_T_SOIL", "MOD_T_MMM", "MOD_P_MMM", "ET_VEG", "K_PLANT", "K_ROOT_STEM", "SIF740"]
    SAVING_DICT[var] = true
end
log("✅ Initialized Configuration and SAVING_DICT")

# Load AVIRIS traits
#batch_file = "data/batches/batch_1_1.nc"
batch_file = "data/test_output_rmse_ndvi.nc"
log("📦 Loading AVIRIS traits from $batch_file")
chls = read_nc(Float64, batch_file, "chl")
lais = read_nc(Float64, batch_file, "lai")
lmas = read_nc(Float64, batch_file, "lma")
lat = read_nc(Float64, batch_file, "lat")
lon = read_nc(Float64, batch_file, "lon")

# Setup SPAC model for Huslia
lat_huslia, lon_huslia = 65.7, -156.4
dict_shift = grid_dict(LandDatasetLabels("gm2", 2020), lat_huslia, lon_huslia)
dict_shift["LONGITUDE"] = lon_huslia
dict_shift["LATITUDE"] = lat_huslia
dict_shift["YEAR"] = 2022
dict_shift["LMA"] = 0.01
dict_shift["soil_color"] = 13
spac_shift = grid_spac(CONFIG, dict_shift)
log("✅ SPAC model initialized")

# Load and Prepare Weather Data
weather_df = grid_weather_driver("wd1", dict_shift)
weather_df.PRECIP .= 0
aviris_date = "2022-07-13T19:05:01.000000"
aviris_date_noon = "2022-07-13T12:00:00.000000"
day_of_year = Dates.dayofyear(DateTime(aviris_date, dateformat"yyyy-mm-ddTHH:MM:SS.ssssss"))
#n = findfirst(weather_df.FDOY .> day_of_year .&& weather_df.RAD .> 300)
#onehour_df = weather_df[n:n, :]


# Filter weather data for the correct day
day_indices = findall(floor.(Int, weather_df.FDOY) .== day_of_year)

if isempty(day_indices)
    error("🚨 No weather data found for day of year: $day_of_year")
end

# Now, explicitly select the timestamp closest to solar noon (maximum RAD)
noon_idx_relative = argmax(weather_df.RAD[day_indices])
n = day_indices[noon_idx_relative]

# Extract your one-hour dataframe at solar noon
onehour_df = weather_df[n:n, :]

#println("🌞 Selected solar noon conditions for FDOY=$(weather_df.FDOY[n]):")
#println("   - RAD=$(weather_df.RAD[n]) W/m²")
#println("   - T_AIR=$(weather_df.T_AIR[n]) K")
#println("   - VPD=$(weather_df.VPD[n]/1000) kPa (converted)")

# Convert VPD from Pa to kPa
onehour_df.VPD .= onehour_df.VPD ./ 1000

# Prepare required MOD_ fields
for (key, save) in SAVING_DICT
    if save
        if key == "MOD_SWC"
            for i in 1:length(spac_shift.soils)
                onehour_df[!, Symbol("MOD_SWC_$i")] .= NaN
            end
        elseif key == "MOD_T_SOIL"
            for i in 1:length(spac_shift.soils)
                onehour_df[!, Symbol("MOD_T_SOIL_$i")] .= NaN
            end
        elseif key == "MOD_T_MMM"
            for label in ["MOD_T_L_MAX", "MOD_T_L_MEAN", "MOD_T_L_MIN"]
                onehour_df[!, Symbol(label)] .= NaN
            end
        elseif key == "MOD_P_MMM"
            for label in ["MOD_P_L_MAX", "MOD_P_L_MEAN", "MOD_P_L_MIN"]
                onehour_df[!, Symbol(label)] .= NaN
            end
        else
            col = Symbol(key)
            if !(col in names(onehour_df))
                onehour_df[!, col] .= NaN
            end
        end
    end
end

@everywhere spac_shift = $spac_shift
@everywhere onehour_df = $onehour_df

@everywhere function run_pixel_simulation(chl, lai, lma, lat, lon)
    if isnan(chl) || isnan(lai) || isnan(lma)
        @info "❌ Skipping NaN traits at (lat=$(lat), lon=$(lon))"
        return NaN
    end

    df = deepcopy(onehour_df)
    df.CHLOROPHYLL .= chl
    df.car .= chl / 7.0
    df.LAI .= lai
    df.VCMAX25 .= 1.30 * chl + 3.72
    df.JMAX25 .= 2.49 * chl + 10.80
    df.LMA .= lma

    df_tuple = NamedTuple{Tuple(Symbol.(names(df)))}(Tuple([df[:, n] for n in names(df)]))

    try
        simulation!(CONFIG, deepcopy(spac_shift), df_tuple; saving_dict=SAVING_DICT)
        gpp = df_tuple.GPP[1]
        @info "✅ Pixel(lat=$(lat), lon=$(lon)): chl=$(chl), lai=$(lai), vcm=$(lma), GPP=$(gpp)"
        @info "🌍 ENV Vars: RAD=$(df.RAD[1]), T_AIR=$(df.T_AIR[1]), SWC=$(df.SWC_1[1]), VPD=$(df.VPD[1]), CO2=$(df.CO2[1]), LAI=$(lai)"
        return gpp
    catch e
        @warn "Simulation error at (lat=$(lat), lon=$(lon)): $e"
        return NaN
    end
end

log("🚀 Running simulations in parallel...")
params = [(chls[i], lais[i], lmas[i], lat[i[1]], lon[i[2]]) for i in CartesianIndices(chls)]
gpp_results = @showprogress pmap(x -> run_pixel_simulation(x...), params)

gpp_matrix = permutedims(reshape(gpp_results, size(chls)), (2, 1))

# Save Results
NCDataset("data/gpp_test_output_rmse_ndvi.nc.nc", "c") do ds
    defDim(ds, "lon", length(lon))
    defDim(ds, "lat", length(lat))

    defVar(ds, "lon", FT, ("lon",))[:] = lon
    defVar(ds, "lat", FT, ("lat",))[:] = lat
    defVar(ds, "GPP", FT, ("lon", "lat"))[:, :] = gpp_matrix

    ds["GPP"].attrib["units"] = "gC.m-2.day-1"
    ds["GPP"].attrib["long_name"] = "Gross Primary Production"
end

log("✅ Results saved")
close(logfile)