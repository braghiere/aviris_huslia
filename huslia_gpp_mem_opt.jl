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
using NetcdfIO: read_nc, create_nc!, append_nc!

# Initialize parallel workers
if nworkers() == 1
    addprocs(48; exeflags="--project")
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

function process_gpp_batch(batch_file::String, output_file::String)
    if isfile(output_file)
        @info "Skipping existing batch: $output_file"
        return
    end

    chls = read_nc(Float64, batch_file, "chl")
    lais = read_nc(Float64, batch_file, "lai")
    lmas = read_nc(Float64, batch_file, "lma")
    lat = read_nc(Float64, batch_file, "lat")
    lon = read_nc(Float64, batch_file, "lon")

    lat_huslia, lon_huslia = 65.7, -156.4
    dict_shift = grid_dict(LandDatasetLabels("gm2", 2020), lat_huslia, lon_huslia)
    dict_shift["LONGITUDE"] = lon_huslia
    dict_shift["LATITUDE"] = lat_huslia
    dict_shift["YEAR"] = 2022
    dict_shift["LMA"] = 0.01
    dict_shift["soil_color"] = 13
    spac_shift = grid_spac(CONFIG, dict_shift)

    weather_df = grid_weather_driver("wd1", dict_shift)
    weather_df.PRECIP .= 0
    doy = dayofyear(DateTime("2022-07-13T12:00:00"))
    day_indices = findall(floor.(Int, weather_df.FDOY) .== doy)
    noon_idx = argmax(weather_df.RAD[day_indices])
    onehour_df = weather_df[day_indices[noon_idx]:day_indices[noon_idx], :]
    onehour_df.VPD ./= 1000

    # --- Critical step you missed: Prepare required MOD_ fields ---
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
            return df_tuple.GPP[1]
        catch e
            @warn "Simulation error at (lat=$(lat), lon=$(lon)): $e"
            return NaN
        end
    end

    params = [(chls[I], lais[I], lmas[I], lat[I[1]], lon[I[2]]) for I in CartesianIndices(chls)]

    gpp_results = @showprogress pmap(x -> run_pixel_simulation(x...), params)
    gpp_matrix = permutedims(reshape(gpp_results, size(chls)), (2, 1))

    create_nc!(output_file, ["lon", "lat"], [length(lon), length(lat)])
    append_nc!(output_file, "lon", lon, Dict("units" => "degrees_east"), ["lon"])
    append_nc!(output_file, "lat", lat, Dict("units" => "degrees_north"), ["lat"])
    append_nc!(output_file, "GPP", gpp_matrix, Dict("units" => "gC.m-2.day-1"), ["lon", "lat"])

    @info "✅ Batch processed and saved: $output_file"
end

batch_dir = "data/batches"
gpp_batch_dir = "data/gpp_batches"
mkpath(gpp_batch_dir)

batch_files = filter(x -> endswith(x, ".nc"), readdir(batch_dir, join=true))

for batch_file in batch_files
    output_file = joinpath(gpp_batch_dir, basename(batch_file))
    process_gpp_batch(batch_file, output_file)
end

@info "✅ All batches processed."