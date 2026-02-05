#!/usr/bin/env julia
# Full domain: Reflectance → Traits → GPP (entire Huslia scene)
# EKI VERSION v4 with:
# - LAI from LiDAR NetCDF (NOT optimized)
# - CI from LiDAR NetCDF (NOT optimized)
# - Only 4 optimized parameters: cab, lwc, cbc, pro
# - Soil moisture = 0.70 (near saturation)

using NetcdfIO: read_nc
using NCDatasets
using Distributed
using ProgressMeter: @showprogress
using Base.GC
using Statistics
using Dates
using JSON3
using LinearAlgebra

println("="^80)
println("Full Domain: Reflectance → Traits → GPP (EKI VERSION v4)")
println("="^80)
println("✨ NEW: LAI & CI from LiDAR, only 4 optimized params, tsm=0.70")

# Configuration
const CHECKPOINT_FILE = "data/processing_checkpoint_eki_v4.json"
const OUTPUT_FILE = "data/output_full_domain_traits_gpp_eki_v4.nc"
const BATCH_SIZE = 500  # Process 500 rows per batch

# Setup parallel workers
if length(workers()) > 1
    rmprocs(workers())
end

num_workers = min(80, Sys.CPU_THREADS - 10)
addprocs(num_workers; exeflags="--project")
println("✅ Using $num_workers parallel workers for EKI v4 (full domain)")

# Load functions on all workers
@everywhere begin
    using NetcdfIO
    using EnsembleKalmanProcesses
    using Emerald
    using Emerald.EmeraldLand.Namespace: BulkSPAC, SPACConfiguration
    using Emerald.EmeraldLand.SPAC: initialize_spac!, prescribe_traits!, read_spectrum, soil_plant_air_continuum!
    using Emerald.EmeraldData.GlobalDatasets: LandDatasetLabels, grid_dict, grid_spac
    using Emerald.EmeraldData.WeatherDrivers: grid_weather_driver
    using Emerald.EmeraldFrontier: simulation!, SAVING_DICT
    using Random
    using LinearAlgebra
    using Statistics
    import Emerald.EmeraldData.WeatherDrivers: DRIVER_FOLDER
    eval(:(Emerald.EmeraldData.WeatherDrivers.DRIVER_FOLDER = "/kiwi-data/Data/model/LAND/drivers"))
    # Setup for trait retrieval
    FT = Float64
    CONFIG_TRAIT = SPACConfiguration(FT)
    CONFIG_TRAIT.ENABLE_SIF = false
    SHIFT_TRAIT = BulkSPAC(CONFIG_TRAIT)
    initialize_spac!(CONFIG_TRAIT, SHIFT_TRAIT)
    SHIFT_BAK = deepcopy(SHIFT_TRAIT)
    const SPECTRA_MIN_WL = minimum(CONFIG_TRAIT.SPECTRA.Λ)
    const SPECTRA_MAX_WL = maximum(CONFIG_TRAIT.SPECTRA.Λ)
    # V4: LAI and CI from LiDAR, only 4 params optimized
    function solver_dict_func(vals::Vector{Float64}, lai_lidar::Float64, ci_lidar::Float64)
        return Dict(
            "cab" => vals[1], 
            "lai" => lai_lidar,  # V4: FROM LIDAR!
            "lwc" => vals[2],
            "cbc" => vals[3], 
            "pro" => vals[4],
            "ci"  => ci_lidar,   # V4: FROM LIDAR!
            "meso_n" => 1.45,    # Fixed for boreal needleleaf
            "sc"  => 13.0,       # Fixed for Huslia (boreal forest)
            "tsm" => 0.70        # V4: Near saturation!
        )
    end
    function target_curve(wls::Vector{FT}, params::Dict{String,FT})::Vector{FT}
        SHIFT = deepcopy(SHIFT_BAK)
        cab_val = get(params, "cab", NaN)
        cbc_val = get(params, "cbc", NaN)
        pro_val = get(params, "pro", NaN)
        lwc_val = get(params, "lwc", NaN)
        meso_n_val = get(params, "meso_n", NaN)
        for leaf in SHIFT.plant.leaves
            if !isnan(meso_n_val)
                leaf.bio.trait.meso_n = meso_n_val
            end
            if !isnan(cab_val)
                leaf.bio.trait.cab = cab_val
                leaf.bio.trait.car = cab_val / 7.0
            end
            if !isnan(cbc_val)
                leaf.bio.trait.cbc = cbc_val
                leaf.bio.trait.lma = leaf.bio.trait.pro + leaf.bio.trait.cbc
            end
            if !isnan(pro_val)
                leaf.bio.trait.pro = pro_val
                leaf.bio.trait.lma = leaf.bio.trait.pro + leaf.bio.trait.cbc
            end
            if !isnan(lwc_val)
                leaf.capacitor.trait.v_max = lwc_val
                leaf.capacitor.state.v_storage = lwc_val
            end
        end
        if haskey(params, "sc")
            SHIFT.soil_bulk.trait.color = Int(round(params["sc"]))
        end
        if haskey(params, "tsm")
            SHIFT.soils[1].state.θ = params["tsm"]
        end
        if haskey(params, "lai")
            lai_val = params["lai"]
            if haskey(params, "ci") && !isnan(params["ci"])
                lai_val = lai_val * params["ci"]
            end
            prescribe_traits!(CONFIG_TRAIT, SHIFT; lai=lai_val)
        end
        initialize_spac!(CONFIG_TRAIT, SHIFT)
        soil_plant_air_continuum!(CONFIG_TRAIT, SHIFT, 0)
        tar_ys = similar(wls)
        @inbounds for i in eachindex(wls)
            wl = wls[i]
            if (SPECTRA_MIN_WL <= wl <= SPECTRA_MAX_WL) && 
               !(1790 <= wl <= 1920) && !(1345 <= wl <= 1415) && !(2380 <= wl <= 2500)
                tar_ys[i] = read_spectrum(CONFIG_TRAIT.SPECTRA.Λ, 
                                          SHIFT.canopy.sensor_geometry.auxil.reflectance, wl)
            else
                tar_ys[i] = FT(NaN)
            end
        end
        return tar_ys
    end
    
    # EKI trait retrieval function - V4 with LAI and CI from LiDAR
    function fit_shift_traits_eki(wavelengths, reflectance, i::Int, j::Int, lai_lidar::Float64, ci_lidar::Float64)
        # Filter valid wavelengths
        valid_mask = .!isnan.(reflectance)
        wavelength_mask = (wavelengths .< 2380) .& 
                          .!(1345 .< wavelengths .< 1415) .& 
                          .!(1790 .< wavelengths .< 1920)
        final_mask = valid_mask .& wavelength_mask
        
        valid_wls = wavelengths[final_mask]
        valid_refl = reflectance[final_mask]
        
        if length(valid_wls) < 50
            return fill(NaN, 7)  # cab, lai, lma, lwc, cbc, pro, ci
        end
        
        # V4: Only 4 retrievable params (cab, lwc, cbc, pro) - LAI and CI from LiDAR
        # Wide bounds (proven best in comparison study)
        num_ensemble = 60  # 15× params
        prior_means = [50.0, 10.0, 0.015, 0.002]  # cab, lwc, cbc, pro
        prior_stds  = [20.0, 5.0, 0.008, 0.0015]  # Permissive variation
        
        rng = MersenneTwister(42)
        ensemble = hcat([randn(num_ensemble) .* prior_stds[i] .+ prior_means[i] 
                 for i in 1:length(prior_means)]...)
        
        # Low noise (0.25% proven best)
        observation_noise = 0.0025
        Γ = Diagonal(observation_noise^2 * ones(length(valid_wls)))
        
        # Prior distributions - V4: Only 4 params
        prior_distributions = [
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("cab", 50.0, 20.0, 5.0, 100.0),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("lwc", 10.0, 5.0, 3.0, 25.0),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("cbc", 0.015, 0.008, 0.001, 0.040),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("pro", 0.002, 0.0015, 0.0001, 0.008)
        ]
        
        prior_combined = EnsembleKalmanProcesses.ParameterDistributions.combine_distributions(prior_distributions)
        
        try
            eki = EnsembleKalmanProcess(ensemble', valid_refl, Γ, Inversion(); 
                                         rng=rng,
                                         failure_handler_method=SampleSuccGauss())
            
            # Run EKI with adaptive stopping (max 15 iterations)
            max_iterations = 15
            prev_loss = Inf
            
            for iter in 1:max_iterations
                # Use get_u_final (raw parameters)
                params_i = EnsembleKalmanProcesses.get_u_final(eki)
                
                G_ensemble = zeros(length(valid_wls), num_ensemble)
                for ens in 1:num_ensemble
                    param_dict = solver_dict_func(params_i[:, ens], lai_lidar, ci_lidar)
                    try
                        G_ensemble[:, ens] = target_curve(valid_wls, param_dict)
                    catch e
                        # Penalize failed runs
                        G_ensemble[:, ens] .= valid_refl .+ 0.1
                    end
                end
                
                # Add jitter to NaN penalties (avoid identical columns)
                for j in 1:num_ensemble
                    if any(isnan, G_ensemble[:, j])
                        G_ensemble[:, j] .= valid_refl .+ 0.1 .+ 1e-4 .* randn(rng, length(valid_refl))
                    end
                end
                
                # Proper termination check
                terminated = EnsembleKalmanProcesses.update_ensemble!(eki, G_ensemble)
                
                # Check convergence
                current_loss = mean(EnsembleKalmanProcesses.get_error(eki)[end])
                if abs(prev_loss - current_loss) / max(prev_loss, 1e-6) < 0.01
                    break
                end
                prev_loss = current_loss
                
                if !isnothing(terminated) && terminated
                    break
                end
            end
            
            
            # Use get_u_mean_final (raw parameters)
            final_params = EnsembleKalmanProcesses.get_u_mean_final(eki)
            
            # V4: Clamp all parameters to be non-negative
            cab_final = max(final_params[1], 0.0)
            lwc_final = max(final_params[2], 0.0)
            cbc_final = max(final_params[3], 0.0)
            pro_final = max(final_params[4], 0.0)
            
            # Compute LMA as cbc + pro
            lma_computed = cbc_final + pro_final
            
            # Return: cab, lai (from LiDAR), lma, lwc, cbc, pro, ci (from LiDAR)
            return [cab_final, lai_lidar, lma_computed, lwc_final, 
                    cbc_final, pro_final, ci_lidar]
        catch e
            println("⚠️  EKI failed at pixel ($i, $j): ", typeof(e))
            return fill(NaN, 7)
        end
    end
end

# Setup GPP configuration on main process
FT = Float64
CONFIG_GPP = SPACConfiguration(FT)
for var in ["GPP"]
    SAVING_DICT[var] = true
end

println("🌍 Setting up SPAC model for Huslia...")
lat_huslia, lon_huslia = 65.7, -156.4
dict_shift = grid_dict(LandDatasetLabels("gm2", 2020), lat_huslia, lon_huslia)
dict_shift["LONGITUDE"] = lon_huslia
dict_shift["LATITUDE"] = lat_huslia
dict_shift["YEAR"] = 2022
dict_shift["LMA"] = 0.01
dict_shift["soil_color"] = 13
spac_shift = grid_spac(CONFIG_GPP, dict_shift)

println("☀️  Loading weather data...")
weather_df = grid_weather_driver("wd1", dict_shift)
weather_df.PRECIP .= 0
aviris_date = "2022-07-13T19:05:01.000000"
day_of_year = Dates.dayofyear(DateTime(aviris_date, dateformat"yyyy-mm-ddTHH:MM:SS.ssssss"))
day_indices = findall(floor.(Int, weather_df.FDOY) .== day_of_year)
noon_idx_relative = argmax(weather_df.RAD[day_indices])
n = day_indices[noon_idx_relative]
onehour_df = weather_df[n:n, :]
onehour_df.VPD .= onehour_df.VPD ./ 1000
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

@everywhere weather_df_global = $onehour_df
@everywhere spac_ref_global = $spac_shift
@everywhere config_global = $CONFIG_GPP
@everywhere saving_dict_global = $SAVING_DICT

# Function to compute traits AND GPP - V4 with LAI/CI from LiDAR
@everywhere function compute_traits_and_gpp(params, weather_df, spac_ref, config, saving_dict, lai_lidar, ci_lidar)
    wavelengths, pixel_ref, i, j = params
    
    # Get traits using EKI (LAI and CI from LiDAR)
    traits = fit_shift_traits_eki(wavelengths, pixel_ref, i, j, lai_lidar, ci_lidar)
    chl, lai, lma, lwc, cbc, pro, ci = traits
    gpp = NaN
    if !isnan(chl) && !isnan(lai) && !isnan(lma)
        df = deepcopy(weather_df)
        df.CHLOROPHYLL .= chl
        df.car .= chl / 7.0
        df.LAI .= lai
        df.VCMAX25 .= 0.8 * (1.30 * chl + 3.72)
        df.JMAX25 .= 0.8 * (2.49 * chl + 10.80)
        df.LMA .= lma
        df_tuple = NamedTuple{Tuple(Symbol.(names(df)))}(Tuple([df[:, n] for n in names(df)]))
        try
            simulation!(config, deepcopy(spac_ref), df_tuple; saving_dict=saving_dict)
            gpp = df_tuple.GPP[1]
        catch e
            println("⚠️  GPP simulation error at pixel ($i, $j): ", typeof(e))
        end
    end
    return (chl, lai, lma, lwc, cbc, pro, ci, gpp)
end

@everywhere function process_pixel(params, lai_lidar, ci_lidar)
    return compute_traits_and_gpp(params, weather_df_global, spac_ref_global, config_global, saving_dict_global, lai_lidar, ci_lidar)
end

function compute_ndvi(reflectance, wavelengths)
    idx_red = findfirst(x -> 660 ≤ x ≤ 680, wavelengths)
    idx_nir = findfirst(x -> 840 ≤ x ≤ 860, wavelengths)
    if idx_red === nothing || idx_nir === nothing
        return NaN
    end
    red = mean(reflectance[idx_red])
    nir = mean(reflectance[idx_nir])
    return (nir - red) / (nir + red + 1e-6)
end

function is_spectrum_valid(reflectance, wavelengths)
    vis_indices = findall(x -> 400 ≤ x ≤ 700, wavelengths)
    nir_indices = findall(x -> 700 ≤ x ≤ 1000, wavelengths)
    if isempty(vis_indices) || isempty(nir_indices)
        return false
    end
    vis_refl = reflectance[vis_indices]
    vis_valid = count(x -> 0 < x < 1, vis_refl)
    vis_fraction = vis_valid / length(vis_refl)
    nir_refl = reflectance[nir_indices]
    nir_valid = count(x -> 0 < x < 1, nir_refl)
    nir_fraction = nir_valid / length(nir_refl)
    return vis_fraction >= 0.5 && nir_fraction >= 0.3
end

# V4: Load LiDAR NetCDF maps
println("\n📊 Loading LiDAR NetCDF maps...")
lai_map = read_nc(Float32, "data/lai_output/netcdf/lai_aviris_grid.nc", "lai")
ci_map = read_nc(Float32, "data/lai_output/netcdf/ci_aviris_grid.nc", "ci")

# Julia reads NetCDF in reverse: Python (lat,lon) → Julia (lon,lat)
println("✓ LAI map size: $(size(lai_map)) (lon, lat)")
println("✓ CI map size: $(size(ci_map)) (lon, lat)")
println("✓ LAI mean: $(round(mean(filter(!isnan, lai_map)), digits=2)) m²/m²")
println("✓ CI mean: $(round(mean(filter(!isnan, ci_map)), digits=2))")

# Helper function to get LiDAR value
function get_lidar_value(lat_idx, lon_idx, data_map)
    if 1 <= lon_idx <= size(data_map, 1) && 1 <= lat_idx <= size(data_map, 2)
        val = data_map[lon_idx, lat_idx]
        return isnan(val) ? nothing : Float64(val)
    end
    return nothing
end

datafile = "data/merged_output_cf.nc"
println("\n📖 Reading $datafile...")
wavelengths = read_nc(Float64, datafile, "wavelength")
reflectance_4d = read_nc(Float64, datafile, "Reflectance")
lat_values = read_nc(Float64, datafile, "lat")
lon_values = read_nc(Float64, datafile, "lon")
reflectance_3d = reflectance_4d[:, :, :, 1]
reflectance_3d[reflectance_3d .< 0] .= NaN
reflectance_3d[reflectance_3d .> 1] .= NaN
full_lon_size, full_lat_size, wave_size = size(reflectance_3d)
println("Full dimensions: lon=$full_lon_size, lat=$full_lat_size, wave=$wave_size")
lon_size, lat_size, wave_size = size(reflectance_3d)
println("Processing dimensions: lon=$lon_size, lat=$lat_size, wave=$wave_size")
println("Total pixels: $(lon_size * lat_size)")
start_row = 1
if isfile(CHECKPOINT_FILE)
    println("\n📋 Found existing checkpoint file!")
    checkpoint = JSON3.read(read(CHECKPOINT_FILE, String))
    start_row = checkpoint.last_completed_row + 1
    println("   Last completed row: $(checkpoint.last_completed_row)")
    println("   Total pixels processed: $(checkpoint.total_pixels_processed)")
    println("   Resuming from row: $start_row")
    if start_row > lat_size
        println("✅ Processing already complete!")
        exit(0)
    end
else
    println("\n🆕 Starting fresh - no checkpoint found")
    println("📝 Creating output NetCDF file: $OUTPUT_FILE")
    ds = NCDataset(OUTPUT_FILE, "c")
    defDim(ds, "time", 1)
    defDim(ds, "lat", lat_size)
    defDim(ds, "lon", lon_size)
    defVar(ds, "time", Float64, ("time",), 
           attrib = Dict("units" => "days since 2000-01-01 00:00:00",
                        "calendar" => "gregorian"))
    ds["time"][:] = [1.0]
    defVar(ds, "lat", Float64, ("lat",),
           attrib = Dict("units" => "degrees_north"))
    ds["lat"][:] = lat_values
    defVar(ds, "lon", Float64, ("lon",),
           attrib = Dict("units" => "degrees_east"))
    ds["lon"][:] = lon_values
    trait_metadata = Dict(
        "ndvi" => ("unitless", "Normalized Difference Vegetation Index"),
        "chl" => ("μg cm⁻²", "Chlorophyll content"),
        "lai" => ("m² m⁻²", "Leaf Area Index (from LiDAR)"),
        "lma" => ("mg cm⁻²", "Leaf Mass per Area"),
        "lwc" => ("mol m⁻²", "Leaf Water Content"),
        "cbc" => ("mg cm⁻²", "Carbon-based constituents"),
        "pro" => ("mg cm⁻²", "Protein"),
        "ci" => ("unitless", "Clumping Index (from LiDAR)"),
        "gpp" => ("μmol CO₂ m⁻² s⁻¹", "Gross Primary Productivity")
    )
    for (varname, (units, longname)) in trait_metadata
        defVar(ds, varname, Float32, ("lon", "lat", "time"),
               fillvalue = NaN32,
               attrib = Dict("units" => units, "long_name" => longname))
        ds[varname][:, :, :] = fill(NaN32, lon_size, lat_size, 1)
    end
    ds.attrib["about"] = "AVIRIS-NG Huslia: Reflectance → Traits → GPP (EKI v4)"
    ds.attrib["created"] = string(now())
    ds.attrib["method"] = "EKI v4: 4 params (cab,lwc,cbc,pro), LAI+CI from LiDAR, tsm=0.70, 60 ensemble, 15 iter"
    ds.attrib["improvements"] = "LAI and CI from LiDAR NetCDF, soil moisture near saturation"
    close(ds)
    println("✅ NetCDF file initialized")
end
total_pixels_processed = isfile(CHECKPOINT_FILE) ? JSON3.read(read(CHECKPOINT_FILE, String)).total_pixels_processed : 0
println("\n🚀 Starting EKI v4 batch processing from row $start_row to $lat_size")
println("   Batch size: $BATCH_SIZE rows")
println("   ✨ v4: LAI+CI from LiDAR, only 4 params, tsm=0.70")
for batch_start in start_row:BATCH_SIZE:lat_size
    batch_end = min(batch_start + BATCH_SIZE - 1, lat_size)
    batch_num = div(batch_start - 1, BATCH_SIZE) + 1
    total_batches = div(lat_size - 1, BATCH_SIZE) + 1
    println("\n" * "="^80)
    println("📦 BATCH $batch_num/$total_batches: Rows $batch_start to $batch_end")
    println("="^80)
    # Find valid pixels with LiDAR data
    println("  🔍 Finding valid pixels with LiDAR...")
    batch_valid_indices = []
    for i in batch_start:batch_end
        for j in 1:lon_size
            pixel_ref = reflectance_3d[j, i, :]
            ndvi = compute_ndvi(pixel_ref, wavelengths)
            
            # V4: Get LiDAR values
            lai_obs = get_lidar_value(i, j, lai_map)
            ci_obs = get_lidar_value(i, j, ci_map)
            
            # Only process if has LiDAR data
            if !isnothing(lai_obs) && !isnothing(ci_obs) && ndvi > 0.2 && is_spectrum_valid(pixel_ref, wavelengths)
                push!(batch_valid_indices, (i, j, lai_obs, ci_obs))
            end
        end
    end
    n_valid = length(batch_valid_indices)
    println("  ✓ Found $n_valid valid pixels with LiDAR")
    if n_valid > 0
        params = [(wavelengths, reflectance_3d[j, i, :], i, j) for (i, j, lai, ci) in batch_valid_indices]
        lidars = [(lai, ci) for (i, j, lai, ci) in batch_valid_indices]
        println("  🚀 Processing $n_valid pixels with EKI v4 (4 params, LAI+CI from LiDAR)...")
        results = @showprogress pmap((p, l) -> process_pixel(p, l[1], l[2]), params, lidars; batch_size=100)
        println("  💾 Writing results...")
        global ds = NCDataset(OUTPUT_FILE, "a")
        for (idx, (i, j, lai_obs, ci_obs)) in enumerate(batch_valid_indices)
            pixel_ref = reflectance_3d[j, i, :]
            ndvi = compute_ndvi(pixel_ref, wavelengths)
            chl, lai, lma, lwc, cbc, pro, ci, gpp = results[idx]
            ds["ndvi"][j, i, 1] = Float32(ndvi)
            ds["chl"][j, i, 1] = Float32(chl)
            ds["lai"][j, i, 1] = Float32(lai)
            ds["lma"][j, i, 1] = Float32(lma)
            ds["lwc"][j, i, 1] = Float32(lwc)
            ds["cbc"][j, i, 1] = Float32(cbc)
            ds["pro"][j, i, 1] = Float32(pro)
            ds["ci"][j, i, 1] = Float32(ci)
            ds["gpp"][j, i, 1] = Float32(gpp)
        end
        close(ds)
        println("  ✅ Results written")
        global total_pixels_processed += n_valid
    else
        println("  ⊘ No valid pixels in this batch")
    end
    checkpoint = Dict(
        "last_completed_row" => batch_end,
        "total_pixels_processed" => total_pixels_processed,
        "timestamp" => string(now()),
        "batch_num" => batch_num,
        "total_batches" => total_batches
    )
    write(CHECKPOINT_FILE, JSON3.write(checkpoint))
    println("  💾 Checkpoint saved (row $batch_end, $total_pixels_processed pixels)")
    GC.gc()
end
println("\n" * "="^80)
println("✅ PROCESSING COMPLETE!")
println("="^80)
println("Total pixels processed: $total_pixels_processed")
println("Output file: $OUTPUT_FILE")
println("Checkpoint file: $CHECKPOINT_FILE")
println("\n📊 Computing summary statistics...")
ds = NCDataset(OUTPUT_FILE, "r")
for varname in ["gpp", "lai", "chl", "ci"]
    data = ds[varname][:, :, 1]
    valid_data = [x for x in data if !ismissing(x) && !isnan(x)]
    if length(valid_data) > 0
        println("  $varname: min=$(round(minimum(valid_data), digits=2)), " *
                "max=$(round(maximum(valid_data), digits=2)), " *
                "mean=$(round(mean(valid_data), digits=2))")
    else
        println("  $varname: No valid values")
    end
end
close(ds)
println("\n🧹 Cleaning up workers...")
if length(workers()) > 1
    rmprocs(workers())
end
println("\n🎉 All done! EKI v4 full domain processing complete.")
println("   Key improvements:")
println("   ✅ LAI and CI from LiDAR NetCDF (not optimized)")
println("   ✅ Only 4 parameters optimized (cab, lwc, cbc, pro)")
println("   ✅ Soil moisture = 0.70 (near saturation)")
println("   ✅ Non-negative parameter clamping")
println("\n   To restart: rm $CHECKPOINT_FILE")
