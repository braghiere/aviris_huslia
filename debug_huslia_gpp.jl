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
logfile = open("fixed_simulation.log", "w")
log(msg) = (println(msg); println(logfile, msg))

# Initialize parallel workers
if nworkers() == 1
    addprocs(4; exeflags="--project")
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
batch_file = "data/batches/batch_1_1.nc"
log("📦 Loading AVIRIS traits from $batch_file")
chls = read_nc(Float64, batch_file, "chl")
lais = read_nc(Float64, batch_file, "lai")
vcms = read_nc(Float64, batch_file, "lma")
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
day_of_year = Dates.dayofyear(DateTime(aviris_date, dateformat"yyyy-mm-ddTHH:MM:SS.ssssss"))

n = findfirst(weather_df.FDOY .> day_of_year .&& weather_df.RAD .> 1)
onehour_df = weather_df[n:n, :]

# ---------------------------------------
# ✅ Prepare onehour_df with required MOD_ fields (corrected)
# ---------------------------------------
n_rows = nrow(onehour_df)

for (key, save) in SAVING_DICT
    if save
        if key == "MOD_SWC"
            for i in 1:length(spac_shift.soils)
                col = Symbol("MOD_SWC_$i")
                onehour_df[!, col] = fill(NaN, n_rows)
            end
        elseif key == "MOD_T_SOIL"
            for i in 1:length(spac_shift.soils)
                col = Symbol("MOD_T_SOIL_$i")
                onehour_df[!, col] = fill(NaN, n_rows)
            end
        elseif key == "MOD_T_MMM"
            for label in ["MOD_T_L_MAX", "MOD_T_L_MEAN", "MOD_T_L_MIN"]
                onehour_df[!, Symbol(label)] = fill(NaN, n_rows)
            end
        elseif key == "MOD_P_MMM"
            for label in ["MOD_P_L_MAX", "MOD_P_L_MEAN", "MOD_P_L_MIN"]
                onehour_df[!, Symbol(label)] = fill(NaN, n_rows)
            end
        else
            col = Symbol(key)
            if !(col in names(onehour_df))
                onehour_df[!, col] = fill(NaN, n_rows)
            end
        end
    end
end

# Distribute shared variables to workers
@everywhere spac_shift = $spac_shift
@everywhere onehour_df = $onehour_df

@everywhere function run_pixel_simulation(chl, lai, vcm)
    df = deepcopy(onehour_df)
    df.CHLOROPHYLL .= chl
    df.car .= chl / 7.0
    df.LAI .= lai
    df.VCMAX25 .= 1.30 * chl + 3.72
    df.JMAX25 .= 2.49 * chl + 10.80
    df.LMA .= vcm

    df_tuple = NamedTuple{Tuple(Symbol.(names(df)))}(Tuple([df[:, n] for n in names(df)]))

    try
        simulation!(CONFIG, deepcopy(spac_shift), df_tuple; saving_dict=SAVING_DICT)
        return df_tuple.GPP[1]
    catch e
        @warn "Simulation error: $e"
        return NaN
    end
end

# Run Simulation in Parallel
log("🚀 Running simulations in parallel...")
params = [(chls[i], lais[i], vcms[i]) for i in eachindex(chls)]
gpp_results = @showprogress pmap(x -> run_pixel_simulation(x...), params)

# Reshape results
gpp_matrix = reshape(gpp_results, size(chls))

# Save Results
output_file = "huslia_simulation_results.nc"
NCDataset(output_file, "c") do ds
    defDim(ds, "lat", length(lat))
    defDim(ds, "lon", length(lon))

    defVar(ds, "lat", FT, ("lat",))[:] = lat
    defVar(ds, "lon", FT, ("lon",))[:] = lon
    defVar(ds, "GPP", FT, ("lat", "lon"))[:, :] = gpp_matrix

    ds["lat"].attrib["units"] = "degrees_north"
    ds["lon"].attrib["units"] = "degrees_east"
    ds["GPP"].attrib["units"] = "gC/m²/day"
    ds["GPP"].attrib["long_name"] = "Gross Primary Production"
end

log("✅ Results saved to $output_file")
close(logfile)
