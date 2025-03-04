using Distributed
using NetcdfIO: append_nc!, create_nc!, read_nc
using ProgressMeter: @showprogress
using Base.GC
using Random
using Statistics
using LinearAlgebra  

# Ensure multiple processes are available for parallel computation
rmprocs(workers())
if length(workers()) == 1
    addprocs(48; exeflags = "--project")  # Adjust process count as needed
end

# Load dependencies on all workers
@everywhere begin
    using EnsembleKalmanProcesses
    using Emerald.EmeraldLand.Namespace: BulkSPAC, SPACConfiguration
    using Emerald.EmeraldLand.SPAC: initialize_spac!, prescribe_traits!, read_spectrum, soil_plant_air_continuum!
    using Emerald.EmeraldMath.Stats: rmse
    using Random
    using LinearAlgebra  

    FT = Float64
    CONFIG = SPACConfiguration(FT)
    CONFIG.ENABLE_SIF = false
    SHIFT = BulkSPAC(CONFIG)
    initialize_spac!(CONFIG, SHIFT)
    SHIFT_BAK = deepcopy(SHIFT)
end

# Read NetCDF data
@everywhere function read_validate_data(datafile::String)
    println("📖 Reading data from file: ", datafile)
    
    # Read data
    wavelengths = read_nc(Float64, datafile, "wavelength")
    reflectance = read_nc(Float64, datafile, "Reflectance")
    lat  = read_nc(Float64, datafile, "lat")  
    lon  = read_nc(Float64, datafile, "lon")  

    println("✅ Data loaded successfully.")
    return wavelengths, reflectance, lat, lon
end

# Select and preprocess pixel data with NaN and wavelength filtering
function select_pixel(reflectance, wavelengths, i::Int, j::Int)
    observed_reflectance = reflectance[i, j, :]

    # Remove NaNs first
    valid_indices = .!isnan.(observed_reflectance)
    
    # Apply wavelength filtering
    wavelength_mask = (wavelengths .< 2380) .& .!(1345 .< wavelengths .< 1415) .& .!(1790 .< wavelengths .< 1920)
    
    # Combine both filters
    final_mask = valid_indices .& wavelength_mask

    valid_wavelengths = wavelengths[final_mask]
    valid_reflectance = observed_reflectance[final_mask]

    if isempty(valid_wavelengths) || isempty(valid_reflectance)
        return nothing  # Skip invalid pixels
    end

    return valid_wavelengths, valid_reflectance, i, j
end

@everywhere function solver_dict_func(vals::Vector{Float64})
    return Dict(
        "cab" => vals[1], "lai" => vals[2], "lma" => vals[3], "lwc" => vals[4],
        "cbc" => vals[5], "pro" => vals[6],
        "sc"  => Float64(Int(round(clamp(vals[7], 1, 20)))),
        "tsm" => vals[8]
    )
end

# -- Forward model
@everywhere function target_curve(wls::Vector{FT}, params::Dict{String,FT})::Vector{FT}
    SHIFT = deepcopy(SHIFT_BAK)

    # Update SHIFT traits
    for key in keys(params)
        for leaf in SHIFT.plant.leaves
            if key == "cab"
                leaf.bio.trait.cab = params["cab"]
            elseif key == "car"
                leaf.bio.trait.car = params["car"]
            elseif key == "cbc"
                leaf.bio.trait.cbc = params["cbc"]
                leaf.bio.trait.lma = leaf.bio.trait.pro + leaf.bio.trait.cbc
            elseif key == "lma"
                leaf.bio.trait.lma = params["lma"]
            elseif key == "pro"
                leaf.bio.trait.pro = params["pro"]
                leaf.bio.trait.lma = leaf.bio.trait.pro + leaf.bio.trait.cbc
            elseif key == "lwc"
                leaf.capacitor.trait.v_max = params["lwc"]
                leaf.capacitor.state.v_storage = params["lwc"]
            end
        end
    end

    # Soil
    if haskey(params, "sc")
        SHIFT.soil_bulk.trait.color = Int(round(params["sc"]))
    end
    if haskey(params, "tsm")
        SHIFT.soils[1].state.θ = params["tsm"]
    end
    if haskey(params, "lai")
        prescribe_traits!(CONFIG, SHIFT; lai=params["lai"])
    end

    initialize_spac!(CONFIG, SHIFT)
    soil_plant_air_continuum!(CONFIG, SHIFT, 0)

    tar_ys = similar(wls)
    min_wl = minimum(CONFIG.SPECTRA.Λ)
    max_wl = maximum(CONFIG.SPECTRA.Λ)

    for i in eachindex(wls)
        wl = wls[i]
        if (min_wl <= wl <= max_wl) && !(1790 <= wl <= 1920) && !(1345 <= wl <=1415) && !(2380<=wl<=2500)
            tar_ys[i] = read_spectrum(CONFIG.SPECTRA.Λ, SHIFT.canopy.sensor_geometry.auxil.reflectance, wl)
        else
            tar_ys[i] = FT(NaN)
        end
    end

    if any(isnan.(tar_ys))
        @warn "target_curve produced NaNs!"
    end

    return tar_ys
end

@everywhere function fit_shift_traits_eki(valid_wavelengths, valid_reflectance, i::Int, j::Int)
    println("\n🚀 Running EKI for pixel ($i, $j)...")

    num_ensemble = 80  # Number of ensemble members - min 10 times num of vars  
    ##############["cab", "lai", "lma", "lwc", "cbc", "pro", "sc", "tsm"]
    prior_means = [40.0, 4.0, 0.012, 10.0, 0.010, 0.0095, 15, 0.5]  
    prior_stds  = [30.0, 2.0, 0.008, 7.0, 0.005, 0.0075, 3, 0.4]   

    rng = MersenneTwister(42)
    ensemble = hcat([randn(num_ensemble) .* prior_stds[i] .+ prior_means[i] for i in 1:length(prior_means)]...)  

    observation_noise = 0.005
    Γ = Diagonal(observation_noise^2 * ones(length(valid_wavelengths)))

    prior_distributions = [
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("cab", prior_means[1], prior_stds[1], 0.1, 80),
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("lai", prior_means[2], prior_stds[2], 0.1, 8.),
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("lma", prior_means[3], prior_stds[3], 1e-6, 0.05),
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("lwc", prior_means[4], prior_stds[4], 0.1, 20),
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("cbc", prior_means[5], prior_stds[5], 0.0005, 0.035),
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("pro", prior_means[6], prior_stds[6], 1e-6, 0.020),
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("sc", prior_means[7], prior_stds[7], 1, 20),
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("tsm", prior_means[8], prior_stds[8], 0.01, 0.99)
    ]
    prior_combined = EnsembleKalmanProcesses.ParameterDistributions.combine_distributions(prior_distributions)

    eki = EnsembleKalmanProcess(ensemble', valid_reflectance, Γ, Inversion(); rng=MersenneTwister(42))

    for step in 1:40
        params_i = EnsembleKalmanProcesses.get_ϕ_final(prior_combined, eki)  # FIXED!

        G_ensemble = hcat([target_curve(valid_wavelengths, solver_dict_func(params_i[:, j])) for j in 1:num_ensemble]...)

        if size(G_ensemble, 1) ≠ size(valid_reflectance, 1)
            println("⚠ Dimension mismatch at ($i, $j), skipping update...")
            continue
        end

        EnsembleKalmanProcesses.update_ensemble!(eki, G_ensemble)
    end

    optimized_params = EnsembleKalmanProcesses.get_ϕ(prior_combined, eki)  # Ensure it uses the correct distribution
    return optimized_params, (i, j)
end


# Process all selected pixels in parallel using `pmap`
function fit_shift_traits_grid!(datafile::String)
    wavelengths, reflectance, lat, lon = read_validate_data(datafile)

    px_range_x = 50:60
    px_range_y = 50:60

    tasks = []
    for i in px_range_x, j in px_range_y
        result = select_pixel(reflectance, wavelengths, i, j)
        if result !== nothing
            push!(tasks, result)
        end
    end

    println("🚀 Running EKI on $(length(tasks)) pixels...")
    fits = @showprogress pmap(x -> fit_shift_traits_eki(x[1], x[2], x[3], x[4]), tasks)
    println("✅ EKI completed for all pixels.")

    pnames = ["cab", "lai", "lma", "lwc", "cbc", "pro", "sc", "tsm"]
    lat_size, lon_size = length(px_range_y), length(px_range_x)

    # Extract results
    println("🔍 Extracting parameter results from EKI fits...")

    param_stats = Dict(
        name => begin
            extracted_means = [mean(f[1][end][ip, :]) for f in fits]  
            extracted_stds  = [std(f[1][end][ip, :]) for f in fits]   
            println("🧐 Extracted $(length(extracted_means)) valid values for $name (mean)")
            println("🧐 Extracted $(length(extracted_stds)) valid values for $name (std)")
            (extracted_means, extracted_stds)
        end
        for (ip, name) in enumerate(pnames)
    )

    println("🌍 Grid dimensions: lat_size = $lat_size, lon_size = $lon_size")

    # Reshape and store parameter maps (both mean and std)
    shaped_mean = Dict(
        name => reshape(param_stats[name][1], lat_size, lon_size)
        for name in pnames
    )
    shaped_std = Dict(
        name => reshape(param_stats[name][2], lat_size, lon_size)
        for name in pnames
    )

    # Convert sc_mean to integer
    shaped_mean["sc"] = round.(Int, shaped_mean["sc"])

    # Create 2D latitude and longitude arrays corresponding to the selected pixels
    lat_grid = reshape([lat[i] for i in px_range_y, j in px_range_x], lat_size, lon_size)
    lon_grid = reshape([lon[j] for i in px_range_y, j in px_range_x], lat_size, lon_size)

    # Define filenames
    ncresult_mean = "data/fitted_traits_eki_mean.nc"
    ncresult_std  = "data/fitted_traits_eki_std.nc"

    # Save MEAN values
    println("📝 Creating NetCDF file for MEANS: $ncresult_mean")
    create_nc!(ncresult_mean, ["lon", "lat"], [lon_size, lat_size])

    println("🗺 Writing actual lat/lon values to MEAN file...")
    append_nc!(ncresult_mean, "lat", lat_grid, Dict("latitude" => "latitude"), ["lon", "lat"])
    append_nc!(ncresult_mean, "lon", lon_grid, Dict("longitude" => "longitude"), ["lon", "lat"])
    
    for name in pnames
        println("📡 Writing $name mean to NetCDF...")
        #append_nc!(ncresult_mean, name*"_mean", shaped_mean[name], Dict(name*"_mean" => name*" mean"), ["lon", "lat"])
        append_nc!(ncresult_mean, name, shaped_mean[name], Dict(name => name), ["lon", "lat"])
    end

    # Save STD (Uncertainty) values
    println("📝 Creating NetCDF file for STD: $ncresult_std")
    create_nc!(ncresult_std, ["lon", "lat"], [lon_size, lat_size])

    println("🗺 Writing actual lat/lon values to STD file...")
    append_nc!(ncresult_std, "lat", lat_grid, Dict("latitude" => "latitude"), ["lon", "lat"])
    append_nc!(ncresult_std, "lon", lon_grid, Dict("longitude" => "longitude"), ["lon", "lat"])


    for name in pnames
        println("📡 Writing $name std (uncertainty) to NetCDF...")
        append_nc!(ncresult_std, name*"_std", shaped_std[name], Dict(name*"_std" => name*" std (uncertainty)"), ["lon", "lat"])
    end

    println("📊 Final extracted parameters:")
    for (k, v) in param_stats
        println("$k => mean length: ", length(v[1]), " | Sample values: ", v[1][1:min(5, length(v[1]))])
        println("$k => std length: ", length(v[2]), " | Sample values: ", v[2][1:min(5, length(v[2]))])
    end

    GC.gc()
    @info "✅ Done! Results saved in: $ncresult_mean (mean) and $ncresult_std (std)"
end

# Run the function
datafile = "data/merged_output_subset.nc"
fit_shift_traits_grid!(datafile)

