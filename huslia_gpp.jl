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

# Set up the number type
FT = Float64

# ---- Optimize Worker Management ----
if length(workers()) > 1
    rmprocs(workers())  # Ensure clean start
end
if length(workers()) == 1
    addprocs(4; exeflags = "--project")  # Adjust number of processes if needed
end

# **🛠️ Load necessary modules on all workers**
@everywhere using Emerald
@everywhere using Emerald.EmeraldData.GlobalDatasets
@everywhere using Emerald.EmeraldData.WeatherDrivers
@everywhere using Emerald.EmeraldLand.Namespace
@everywhere using Emerald.EmeraldFrontier: simulation!, SAVING_DICT

# Ensure `linear_θ_soil` is defined on all workers
@everywhere linear_θ_soil(x) = min(1, max(eps(), (x - 0.034) / (0.46 - 0.034)))

# Define the function outside of @everywhere but ensure workers have access
@everywhere function run_shift_simulation!(param::Vector)
    if any(isnothing, param)
        return NaN, NaN, NaN, NaN, NaN, NaN
    end

    config = param[1]
    spac = deepcopy(param[2])
    df = param[3]

    # Convert DataFrame to NamedTuple
    df_tuple = (; (Symbol(col) => collect(df[!, col]) for col in propertynames(df))...)

    # **DEBUG: Check for missing fields**
    debug_namedtuple_structure!(df_tuple)

    # Ensure required variables exist as vectors
    required_vars = [
        :MOD_SWC_1, :MOD_SWC_2, :MOD_SWC_3, :MOD_SWC_4,
        :MOD_T_SOIL_1, :MOD_T_SOIL_2, :MOD_T_SOIL_3, :MOD_T_SOIL_4,
        :MOD_T_L_MAX,  # 🌟 NEWLY HANDLED FIELD
        :F_GPP, :SIF740, :SIF683, :SIF757, :SIF771, :F_H2O
    ]

    for var in required_vars
        if !haskey(df_tuple, var)
            println("⚠️ Adding missing field: $var with NaN values")
            df_tuple = merge(df_tuple, NamedTuple{(var,)}((fill(NaN, 1),)))  # Assign missing vars as Vector{Float64}
        elseif df_tuple[var] isa Float64
            df_tuple = merge(df_tuple, NamedTuple{(var,)}((fill(df_tuple[var], 1),)))  # Convert Float64 to vector
        end
    end

    # Run the simulation with corrected NamedTuple
    simulation!(config, spac, df_tuple; saving_dict=SAVING_DICT)

    # Release memory
    spac = nothing
    GC.gc()

    return df_tuple.F_GPP[1], df_tuple.SIF740[1], df_tuple.SIF683[1], df_tuple.SIF757[1], df_tuple.SIF771[1], df_tuple.F_H2O[1]
end



@everywhere function debug_namedtuple_structure!(df_tuple)
    println("🔍 Checking df_tuple structure on Worker $(myid())...")
    println("📋 Available keys in df_tuple: ", keys(df_tuple))

    # Define the required fields for the simulation
    required_fields = [
        :MOD_SWC_1, :MOD_SWC_2, :MOD_SWC_3, :MOD_SWC_4,  # Soil moisture levels
        :MOD_T_SOIL_1, :MOD_T_SOIL_2, :MOD_T_SOIL_3, :MOD_T_SOIL_4,  # Soil temperatures
        :MOD_T_L_MAX,  # 🌟 THIS IS THE NEWLY IDENTIFIED MISSING FIELD
        :F_GPP, :SIF740, :SIF683, :SIF757, :SIF771, :F_H2O  # Model outputs
    ]

    # Check for missing fields
    missing_fields = []
    for field in required_fields
        if !haskey(df_tuple, field)
            push!(missing_fields, field)
        end
    end

    if isempty(missing_fields)
        println("✅ All required fields are present.")
    else
        println("🚨 Missing fields in df_tuple: ", missing_fields)
    end
end






@info "Defining a configuration for the simulation..."
CONFIG = SPACConfiguration(Float64)

# Set the location for Huslia (based on AVIRIS flight region)
huslia_lat = 65.7
huslia_lon = -156.4

@info "Reading the traits data from AVIRIS batch file..."
batch_file = "data/batches/batch_1_1.nc"

# Read new trait data from the batch file
chls = read_nc(FT, batch_file, "chl")
lais = read_nc(FT, batch_file, "lai")
vcms = read_nc(FT, batch_file, "lma")

# Read latitude and longitude from the batch file
lat_values = read_nc(FT, batch_file, "lat")
lon_values = read_nc(FT, batch_file, "lon")

@info "Fetching environmental data for Huslia..."
dict_shift = grid_dict(LandDatasetLabels("gm2", 2020), huslia_lat, huslia_lon)
dict_shift["LONGITUDE"] = huslia_lon
dict_shift["LMA"] = 0.01
dict_shift["soil_color"] = 13
dict_shift["SOIL_N"] = [1.37 for _ in 1:4]
dict_shift["SOIL_α"] = [163.2656 for _ in 1:4]
dict_shift["SOIL_ΘR"] = [0.034 for _ in 1:4]
dict_shift["SOIL_ΘS"] = [0.46 for _ in 1:4]

@info "Initializing SPAC model..."
spac_shift = grid_spac(CONFIG, dict_shift)
g1 = dict_shift["G1_MEDLYN_C3"]
bt = BetaFunction{Float64}(FUNC = linear_θ_soil, PARAM_X = BetaParameterΘ(), PARAM_Y = BetaParameterG1())

for leaf in spac_shift.plant.leaves
    leaf.flux.trait.stomatal_model = MedlynSM{Float64}(G0 = 0.005, G1 = g1, β = bt)
end

@info "Reading weather data for Huslia..."
dict_shift["YEAR"] = 2022
wdrv_shift = grid_weather_driver("wd1", dict_shift)
wdrv_shift.PRECIP .= 0

"""
    run_one_timestep!(day::Int, chls::Matrix, lais::Matrix, vcms::Matrix)
Runs the photosynthesis and SIF simulation for a given day using input traits.
"""
function run_one_timestep!(day::Int, chls::Matrix, lais::Matrix, vcms::Matrix)
    @assert size(chls) == size(lais) == size(vcms) "Mismatch in input sizes!"

    n = findfirst(wdrv_shift.FDOY .> day .&& wdrv_shift.RAD .> 1)
    oneday_df = wdrv_shift[n:n+23,:]
    _, m = findmax(oneday_df.RAD)
    onehour_df = oneday_df[m:m,:]

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

    # Use `pmap` for parallel execution
    results = @showprogress pmap(run_shift_simulation!, params)

    gpps = [r[1] for r in results]
    sifs740 = [r[2] for r in results]
    sifs683 = [r[3] for r in results]
    sifs757 = [r[4] for r in results]
    sifs771 = [r[5] for r in results]
    transps = [r[6] for r in results]

    GC.gc()

    return reshape(gpps, size(chls,1), size(chls,2)), 
           reshape(sifs740, size(chls,1), size(chls,2)), 
           reshape(sifs683, size(chls,1), size(chls,2)), 
           reshape(sifs757, size(chls,1), size(chls,2)), 
           reshape(sifs771, size(chls,1), size(chls,2)), 
           reshape(transps, size(chls,1), size(chls,2))
end

@info "Running the simulation..."

# **Select the AVIRIS Flight Date**
aviris_date = "2022-07-13T19:05:01.000000"
day_of_year = Dates.dayofyear(Dates.DateTime(aviris_date, "yyyy-mm-ddTHH:MM:SS.ssssss"))
@info "Processing data for AVIRIS flight day (DOY: $day_of_year)..."

gpps, sifs740, sifs683, sifs757, sifs771, transps = run_one_timestep!(day_of_year, chls, lais, vcms)

@info "Results for Huslia (AVIRIS flight day) saved successfully!"
