using NetcdfIO: append_nc!, create_nc!, read_nc
using Distributed
using ProgressMeter: @showprogress
using Emerald
using Emerald.EmeraldData.GlobalDatasets: LandDatasetLabels, grid_dict, grid_spac
using Emerald.EmeraldData.WeatherDrivers: grid_weather_driver
using Emerald.EmeraldLand.Namespace: SPACConfiguration, BetaFunction, BetaParameterΘ, BetaParameterG1, MedlynSM
using Emerald.EmeraldFrontier: simulation!, SAVING_DICT
using Dates
using NetCDF
using Base.GC
using Statistics: mean
using NetCDF  # ✅ Fix missing import


#println("📌 SAVING_DICT: ", SAVING_DICT)
#exit(0)   # Normal exit (Success)


FT = Float64

# ---- Optimize Worker Management ----
if length(workers()) > 1
    rmprocs(workers())  # Ensure clean start
end
if length(workers()) == 1
    addprocs(4; exeflags = "--project")  # Adjust number of processes if needed
end

# **🛠️ Load necessary modules on all workers**
@everywhere using DataFrames  # ✅ Load DataFrames on all workers
@everywhere using Emerald
@everywhere using Emerald.EmeraldData.GlobalDatasets
@everywhere using Emerald.EmeraldData.WeatherDrivers
@everywhere using Emerald.EmeraldLand.Namespace
@everywhere using Emerald.EmeraldFrontier: simulation!, SAVING_DICT
@everywhere using Statistics: mean

# ✅ **Ensure Safe Computation of Max without NaN Issues**
@everywhere function safe_max(arr)
    valid_values = filter(!isnan, arr)
    return isempty(valid_values) ? NaN : maximum(valid_values)
end

# ✅ **Debugging Helper Functions**
@everywhere function debug_df(df)
    println("🔍 Checking DataFrame integrity...")
    println("Columns in df: ", names(df))
    println("Column types: ", Dict(col => eltype(df[!, col]) for col in names(df)))
    println("First 5 rows:")
    display(df[1:min(5, size(df, 1)), :])
end

@everywhere function preprocess_df(df)
    println("\n🔍 Converting DataFrame to NamedTuple...")
    
    #df = DataFrame(df)  # Ensure mutability
    df = deepcopy(DataFrame(df))

    
    # 🔹 Print before renaming
    println("📌 Before renaming, first row: ", df[1, :])

    field_mapping = Dict(
        :SWC_1 => :MOD_SWC_1, :SWC_2 => :MOD_SWC_2, :SWC_3 => :MOD_SWC_3, :SWC_4 => :MOD_SWC_4,
        :T_SOIL_1 => :MOD_T_SOIL_1, :T_SOIL_2 => :MOD_T_SOIL_2, :T_SOIL_3 => :MOD_T_SOIL_3, :T_SOIL_4 => :MOD_T_SOIL_4
    )

    for (old_key, new_key) in field_mapping
        old_key_str = string(old_key)  # Convert to string for comparison
        if old_key_str in names(df)
            df[!, new_key] = df[!, Symbol(old_key_str)]  # Convert back to Symbol for access
            println("✅ Copied $old_key → $new_key, first value: ", df[1, new_key])
        else
            println("⚠️ $old_key is missing! Initializing $new_key with NaN.")
            df[!, new_key] .= NaN
        end
    end

    # ✅ Compute actual T_LEAF_MAX, T_LEAF_MIN, and T_LEAF_MEAN
    if :T_LEAF in names(df)
        df[!, :MOD_T_L_MAX] = maximum(df.T_LEAF)  # Get max over time
        df[!, :MOD_T_L_MIN] = minimum(df.T_LEAF)  # Get min over time
        df[!, :MOD_T_L_MEAN] = mean(df.T_LEAF)    # Get mean over time
        println("✅ Computed MOD_T_L_MAX, MOD_T_L_MIN, MOD_T_L_MEAN from T_LEAF values.")
    else
        println("🚨 ERROR: `T_LEAF` is missing! Initializing leaf temperatures with NaN.")
        df[!, :MOD_T_L_MAX] .= 273.15 + 35  # Default to 0°C
        df[!, :MOD_T_L_MIN] .= 273.15 + 15
        df[!, :MOD_T_L_MEAN] .= 273.15 + 25
    end

    # 🔹 Print after renaming
    println("📌 After renaming, first row: ", df[1, :])

    # Convert DataFrame to NamedTuple
    return NamedTuple{Tuple(Symbol.(names(df)))}(Tuple([df[:,n] for n in names(df)])) 
end


# ✅ **Run Debug Simulation**
@everywhere function run_debug_simulation!(param::Vector)
    if any(isnothing, param)
        return NaN, NaN, NaN, NaN, NaN, NaN
    end

    config = param[1]
    spac = deepcopy(param[2])
    df = param[3]

    println("\n🔍 Checking DataFrame before conversion...")
    debug_df(df)

    df_tuple = preprocess_df(df)  # Use new preprocessing function

    # 🔹 Ensure SWC & T_SOIL fields exist before simulation
    println("\n📌 Checking MOD_SWC_1 before simulation: ", df_tuple.MOD_SWC_1[1])
    println("📌 Checking MOD_T_SOIL_1 before simulation: ", df_tuple.MOD_T_SOIL_1[1])

    println("📌 Checking MOD_T_SOIL_1 before simulation: ", df_tuple.MOD_T_SOIL_1[1])

    println("\n🚀 Running simulation!...")
    try
        simulation!(config, spac, df_tuple; saving_dict=SAVING_DICT)

        println("✅ simulation! ran successfully!")
    catch e
        println("🚨 Error in simulation!: ", e)
        return NaN, NaN, NaN, NaN, NaN, NaN
    end

    # Convert back to DataFrame after simulation
    df_result = DataFrame(df_tuple)
    println("🔍 Checking Simulation Outputs: ", names(df_result))
    exit(0)

    # Print all available fields for debugging
    println("🔍 Available fields in df_result after simulation!: ", names(df_result))

    # Check if `F_GPP` exists
    if !(:F_GPP in names(df_result))
        println("🚨 ERROR: `F_GPP` is missing after simulation! Returning NaN.")
        return NaN, NaN, NaN, NaN, NaN, NaN
    end

    GC.gc()
    println("`F_GPP` is missing after simulation! Returning NaN.", df_result.GPP[1])

    return df_result.GPP[1], df_result.SIF740[1], df_result.SIF683[1], df_result.SIF757[1], df_result.SIF771[1], df_result.F_H2O[1]
end





# **🌍 Setup Huslia Data**
CONFIG = SPACConfiguration(Float64)
huslia_lat = 65.7
huslia_lon = -156.4

@info "Reading the traits data from AVIRIS batch file..."
batch_file = "data/batches/batch_1_1.nc"

chls = read_nc(FT, batch_file, "chl")
lais = read_nc(FT, batch_file, "lai")
vcms = read_nc(FT, batch_file, "lma")
lat = read_nc(FT, batch_file, "lat")
lon = read_nc(FT, batch_file, "lon")

@info "Fetching environmental data for Huslia..."
dict_shift = grid_dict(LandDatasetLabels("gm2", 2020), huslia_lat, huslia_lon)
dict_shift["LONGITUDE"] = huslia_lon
dict_shift["LMA"] = 0.01
dict_shift["soil_color"] = 13

@info "Initializing SPAC model..."
spac_shift = grid_spac(CONFIG, dict_shift)

@info "Reading weather data for Huslia..."
dict_shift["YEAR"] = 2022
wdrv_shift = grid_weather_driver("wd1", dict_shift)
wdrv_shift.PRECIP .= 0

@info "Running the simulation..."
aviris_date = "2022-07-13T19:05:01.000000"
day_of_year = Dates.dayofyear(Dates.DateTime(aviris_date, "yyyy-mm-ddTHH:MM:SS.ssssss"))
@info "Processing data for AVIRIS flight day (DOY: $day_of_year)..."

@everywhere using Emerald
@everywhere using Emerald.EmeraldFrontier  # ✅ Now at top level!

function run_debug_timestep!(day::Int, chls::Matrix, lais::Matrix, vcms::Matrix)
    n = findfirst(wdrv_shift.FDOY .> day .&& wdrv_shift.RAD .> 1)
    oneday_df = wdrv_shift[n:n+23, :]
    _, m = findmax(oneday_df.RAD)
    onehour_df = oneday_df[m:m, :]

    params = []
    for i in eachindex(chls)
        if any(isnan, (chls[i], lais[i], vcms[i]))
            push!(params, [nothing, nothing, nothing])
        else
            df = deepcopy(onehour_df)
            df.CHLOROPHYLL .= chls[i]
            df.car .= chls[i] / 7.0
            df.LAI .= lais[i]
            df.VCMAX25 .= 1.30 * chls[i] .+ 3.72
            df.JMAX25 .= 2.49 * chls[i] .+ 10.80
            df.LMA .= vcms[i]

            param = [CONFIG, spac_shift, df]
            push!(params, param)
        end
    end

    println("\n🔍 Running `pmap` with `run_debug_simulation!`...")
    results = @showprogress pmap(run_debug_simulation!, params)
    println("\n✅ `pmap` completed successfully!")

    return reshape([r[1] for r in results], length(lat), length(lon))
end



# Run the simulation
gpp_matrix = run_debug_timestep!(day_of_year, chls, lais, vcms)

# 🚨 Check if all GPP values are NaN
if all(isnan, gpp_matrix)
    println("🚨🚨 ERROR: All GPP values are NaN! Debug inputs and simulation!")
end

# **✅ Save results to NetCDF**
output_file = "debug_huslia_results.nc"

@info "📡 Writing results to NetCDF file: $output_file"
NCDataset(output_file, "c") do ds
    defDim(ds, "lat", length(lat))
    defDim(ds, "lon", length(lon))

    lat_var = defVar(ds, "lat", FT, ("lat",))
    lon_var = defVar(ds, "lon", FT, ("lon",))
    gpp_var = defVar(ds, "GPP", FT, ("lat", "lon"))

    lat_var[:] = lat
    lon_var[:] = lon
    gpp_var[:] = gpp_matrix

    lat_var.attrib["units"] = "degrees_north"
    lon_var.attrib["units"] = "degrees_east"
    gpp_var.attrib["units"] = "gC/m²/day"
    gpp_var.attrib["long_name"] = "Gross Primary Production"
end

@info "✅ Debugging completed! Output saved to: $output_file"
