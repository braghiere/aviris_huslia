using Distributed
using NetcdfIO: append_nc!, create_nc!, read_nc
using ProgressMeter: @showprogress
using Base.GC
using Printf
using Random
using Statistics
using LinearAlgebra  

# Ensure multiple processes are available for parallel computation
rmprocs(workers())
if length(workers()) == 1
    addprocs(8; exeflags = "--project")
end

# Load dependencies on all workers
@everywhere begin
    using EnsembleKalmanProcesses
    using Emerald.EmeraldLand.Namespace: BulkSPAC, SPACConfiguration
    using Emerald.EmeraldLand.SPAC: initialize_spac!, prescribe_traits!, read_spectrum, soil_plant_air_continuum!
    using Emerald.EmeraldMath.Stats: rmse
    using NetcdfIO: append_nc!, create_nc!, read_nc
    using ProgressMeter: @showprogress
    using Random
    using LinearAlgebra  

    FT = Float64
    CONFIG = SPACConfiguration(FT)
    CONFIG.ENABLE_SIF = false
    SHIFT = BulkSPAC(CONFIG)
    initialize_spac!(CONFIG, SHIFT)
    SHIFT_BAK = deepcopy(SHIFT)

    # Ensure initialize_ensemble is defined
    @everywhere function initialize_ensemble(num_ensemble::Int, prior_means::Vector{FT}, prior_stds::Vector{FT}, min_vals::Vector{FT}, max_vals::Vector{FT}, perturbation::FT)
        rng = MersenneTwister(42)
    
        # Ensure correct matrix shape: (num_params, num_ensemble)
        ensemble = transpose(hcat([randn(num_ensemble) .* prior_stds[i] .+ prior_means[i] for i in 1:length(prior_means)]...))
    
        println("✅ Initialized ensemble with size: ", size(ensemble))
    
        # Rescale the ensemble values to be within the specified ranges for each parameter
        for i in 1:size(ensemble, 1)  # Now iterating over parameters
            min_val, max_val = min_vals[i], max_vals[i]
    
            # Avoid division by zero if all values in row are equal
            range_diff = maximum(ensemble[i, :]) - minimum(ensemble[i, :])
            if range_diff > 1e-10
                ensemble[i, :] .= min_val .+ (max_val - min_val) .* (ensemble[i, :] .- minimum(ensemble[i, :])) ./ range_diff
            else
                println("⚠️ Warning: Zero range at parameter ", i, " using mean value.")
                ensemble[i, :] .= mean([min_val, max_val])  # Default to mean
            end
        end
    
        # Add perturbations
        ensemble .+= perturbation .* randn(size(ensemble, 1), size(ensemble, 2))
    
        return ensemble
    end
    

    println("✅ Function initialize_ensemble loaded on worker ", myid())

    # Target function: now filters wavelengths dynamically per (i, j)
    @everywhere function target_curve(ref_x::Vector{FT}, params::Dict{String, FT}) where {FT}
        SHIFT = deepcopy(SHIFT_BAK)

        # Update plant traits
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

        # Update soil properties
        if "sc" in keys(params)
            SHIFT.soil_bulk.trait.color = Int(round(params["sc"]))
        end
        if "tsm" in keys(params)
            SHIFT.soils[1].state.θ = params["tsm"]
        end

        # Update LAI
        if "lai" in keys(params)
            prescribe_traits!(CONFIG, SHIFT; lai = params["lai"])
        end

        initialize_spac!(CONFIG, SHIFT)
        soil_plant_air_continuum!(CONFIG, SHIFT, 0)

        _tar_ys = similar(ref_x)
        _min_wl = minimum(CONFIG.SPECTRA.Λ)
        _max_wl = maximum(CONFIG.SPECTRA.Λ)
        for i in eachindex(ref_x)
            if (_min_wl <= ref_x[i] <= _max_wl) && !(1790 <= ref_x[i] <= 1920) && !(1345 <= ref_x[i] <= 1415) && !(2380 <= ref_x[i] <= 2500) 
                _tar_ys[i] = read_spectrum(CONFIG.SPECTRA.Λ, SHIFT.canopy.sensor_geometry.auxil.reflectance, ref_x[i])
            else
                _tar_ys[i] = FT(NaN)
            end
        end


        # Ensure NaNs are removed dynamically
        #valid_indices = .!isnan.(_tar_ys)
        return _tar_ys
    end
end

@everywhere function solver_dict_func(vals::Vector{Float64})
    return Dict(
        "cab" => vals[1],
        "lai" => vals[2],
        "lma" => vals[3],
        "lwc" => vals[4],
        "cbc" => vals[5],
        "pro" => vals[6],
        "sc"  => Float64(Int(round(clamp(vals[7], 1, 20)))), # Ensure integer for soil color
        "tsm" => vals[8]
    )
end

# Function to fit traits using EKI with dynamic wavelength filtering per (i, j)
@everywhere function fit_shift_traits_eki(wavelengths::Vector{Float64},
        reflectance::Vector{Float64},
        i::Int, j::Int)
    println("Running fit_shift_traits_eki on worker ", myid(), " for pixel ($i, $j)")

    # 1) filter out the invalid bits here, so it’s consistent the whole time
    valid_mask = .!isnan.(reflectance)
    if !any(valid_mask)
        return fill(NaN, 8), (i, j)
    end

    reflectance_filt  = reflectance[valid_mask]               # length = N_obs
    wavelengths_filt  = wavelengths[valid_mask]               # same length

    #println(" wavelengths_filt($i,$j): ", wavelengths_filt)
    #println(" reflectance_filt($i,$j): ", reflectance_filt)

    # 3) Prepare prior and ensemble
    num_ensemble = 80  
    prior_means = [40.0, 3.0, 0.012, 7.5, 0.010, 0.0095, 17.0, 0.5]  
    prior_stds  = [30.0, 1.5, 0.007, 5.0, 0.005, 0.0025, 2.0, 0.4]   
    ensemble = initialize_ensemble(
    num_ensemble, prior_means, prior_stds,
    [0.01, 0.01, 1e-5, 0.1, 0.0005, 1e-5, 1, 0.01], 
    [80.0, 7.00, 0.05, 20.0, 0.035, 0.020, 20, 0.99],
    0.1
    )

    observation_noise = 0.005
    #Γ = Diagonal(observation_noise^2 * ones(length(wavelengths)))
    Γ = Diagonal(observation_noise^2 * ones(length(wavelengths_filt)))

    # 4) Define parameter distributions
    prior_distributions = [
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("cab", prior_means[1], prior_stds[1], 0.01, 80),
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("lai", prior_means[2], prior_stds[2], 0.01, 7),
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("lma", prior_means[3], prior_stds[3], 1e-5, 0.05),
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("lwc", prior_means[4], prior_stds[4], 0.1, 20),
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("cbc", prior_means[5], prior_stds[5], 0.0005, 0.035),
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("pro", prior_means[6], prior_stds[6], 1e-5, 0.020),
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("sc",  prior_means[7], prior_stds[7], 1, 20),
        EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("tsm", prior_means[8], prior_stds[8], 0.01, 0.99)
    ]
    prior_combined = EnsembleKalmanProcesses.ParameterDistributions.combine_distributions(prior_distributions)

    # 5) Initialize EKI
    optimized_params = fill(NaN, 8)
    try
    eki = EnsembleKalmanProcess(ensemble, reflectance_filt, Γ, Inversion(); rng=MersenneTwister(42))

    # 6) EKI iterations
    for step in 1:20
        # Get current param matrix, shape = (num_params, num_ensemble)
        params_i = EnsembleKalmanProcesses.get_ϕ_final(prior_combined, eki)

        # Simulate reflectance for each ensemble member
        simulated_list = [
        target_curve(wavelengths_filt, solver_dict_func(params_i[:, j]))
        for j in 1:num_ensemble
        ]
        #println(" simulated_list[1] : ", simulated_list[1])
        #println(" reflectance_filt : ", reflectance_filt, length(reflectance_filt))
        #exit()


        # Each simulated_list[j] is now length == length(wavelengths).


        obs_filtered = reflectance_filt
        #G_ensemble   = hcat([sim[combined_mask] for sim in simulated_list]...)
        G_ensemble = hcat(simulated_list...)

        #combined_mask = .!isnan.(reflectance_filt)
        #for sim in simulated_list
        #     combined_mask .&= .!isnan.(sim)
        #end
        #
        #obs_filtered = reflectance_filt[combined_mask]
        #G_ensemble   = hcat([sim[combined_mask] for sim in simulated_list]...)


        if size(G_ensemble, 1) != length(obs_filtered)
            @warn "Dimension mismatch at pixel ($i,$j). Skipping EKI update."
            continue
        end
        #EnsembleKalmanProcesses.update_ensemble!(eki, G_ensemble; y=obs_filtered)
        EnsembleKalmanProcesses.update_ensemble!(eki, G_ensemble)
    end

    # 9) Extract final parameter set
    optimized_params = EnsembleKalmanProcesses.get_ϕ(prior_combined, eki)

    catch e
    println("⚠️ EKI error at pixel ($i,$j): ", e)
    end


    return optimized_params, (i, j)
end



# Main function
@everywhere function fit_shift_traits_grid!(datafile::String, ncresult::String)
    println("📖 Reading data from file: ", datafile)

    # ✅ Read wavelength, reflectance, latitude, and longitude
    wavelengths = read_nc(Float64, datafile, "wavelength")
    reflectance = read_nc(Float64, datafile, "Reflectance")
    lat_values  = read_nc(Float64, datafile, "lat")  
    lon_values  = read_nc(Float64, datafile, "lon")  

    # ✅ Clamp reflectance values to ensure valid values
    reflectance = clamp.(reflectance, 0.0, 1.0)

    # ✅ Define the region of interest
    pixel_range_x = axes(reflectance, 1)[50:51]
    pixel_range_y = axes(reflectance, 2)[50:51]

    params = []
    #push!(params, (wavelengths, reflectance[i, j, :], i, j))
    for i in pixel_range_x, j in pixel_range_y
        # Build a combined mask that excludes NaNs AND includes only wl < 2400 nm
        valid_indices = .!isnan.(reflectance[i, j, :]) .& 
        (wavelengths .< 2380.0) .&
        .!(1345 .< wavelengths .< 1415) .&
        .!(1790 .< wavelengths .< 1920)
        if any(valid_indices)
            push!(params, (wavelengths[valid_indices], reflectance[i, j, valid_indices], i, j))
        end
    end


    println("🚀 Running parallel fitting on $(length(params)) valid pixels...")
    fittings = @showprogress pmap(x -> fit_shift_traits_eki(x[1], x[2], x[3], x[4]), params)

    println("💾 Saving results to: ", ncresult)

    # ✅ Extract fitted parameters (means and std deviations)
    param_names = ["cab", "lai", "lma", "lwc", "cbc", "pro", "sc", "tsm"]

    mean_params = Dict(name => [mean(f[1][i, :]) for f in fittings] for (i, name) in enumerate(param_names))
    std_params  = Dict(name => [std(f[1][i, :]) for f in fittings] for (i, name) in enumerate(param_names))

    lat_size = length(pixel_range_y)
    lon_size = length(pixel_range_x)

    # ✅ Reshape to match the NetCDF spatial structure
    reshaped_means = Dict(name => reshape(mean_params[name], lat_size, lon_size) for name in param_names)
    reshaped_stds  = Dict(name => reshape(std_params[name], lat_size, lon_size) for name in param_names)

    # ✅ Create NetCDF and store results
    create_nc!(ncresult, ["lon", "lat"], [lon_size, lat_size])  
    append_nc!(ncresult, "lat", lat_values[pixel_range_y], Dict("latitude" => "latitude"), ["lat"])
    append_nc!(ncresult, "lon", lon_values[pixel_range_x], Dict("longitude" => "longitude"), ["lon"])

    # ✅ Store parameter means and std deviations
    for name in param_names
        append_nc!(ncresult, name * "_mean", reshaped_means[name], Dict(name * "_mean" => name * " mean"), ["lon", "lat"])
        append_nc!(ncresult, name * "_std", reshaped_stds[name], Dict(name * "_std" => name * " standard deviation"), ["lon", "lat"])
    end

    # Display results
    # Display results
    for (param, values) in mean_params
        println("📊 $param mean values:\n", values)
    end

    GC.gc()
    println("✅ Fitting complete. Results saved in: ", ncresult)
end

# Run the function
datafile = "merged_output_subset.nc"
ncresult = "test_fitted_traits_eki.nc"
fit_shift_traits_grid!(datafile, ncresult)

