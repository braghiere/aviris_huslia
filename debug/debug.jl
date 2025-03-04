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
    addprocs(1; exeflags = "--project")
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

# Debugging Checkpoint Function
function user_continue(step::String)
    println("\n🔎 DEBUG CHECKPOINT: $step")
    println("➡ Press Enter to continue, or type 'exit' to stop debugging.")
    input = readline()
    if input == "exit"
        error("❌ Debugging stopped by user at step: $step")
    end
end

# Load NetCDF data
function read_validate_data(datafile::String)
    println("📖 Reading data from file: ", datafile)
    
    # Read data
    wavelengths = read_nc(Float64, datafile, "wavelength")
    reflectance = read_nc(Float64, datafile, "Reflectance")
    lat_values  = read_nc(Float64, datafile, "lat")  
    lon_values  = read_nc(Float64, datafile, "lon")  

    # Print basic info
    println("✅ Data loaded successfully.")
    println("🔹 Wavelengths: ", length(wavelengths))
    println("🔹 Reflectance shape: ", size(reflectance))
    println("🔹 Lat/Lon shape: ", size(lat_values), " / ", size(lon_values))

    user_continue("Loaded and validated NetCDF data")

    return wavelengths, reflectance, lat_values, lon_values
end

# Select a pixel
function select_pixel(reflectance, wavelengths, i::Int, j::Int)
    println("\n📌 Selecting pixel ($i, $j)")

    observed_reflectance = reflectance[i, j, :]
    println("🔎 Raw reflectance values: ", observed_reflectance)

    # Remove NaNs
    valid_indices = .!isnan.(observed_reflectance)
    valid_wavelengths = wavelengths[valid_indices]
    valid_reflectance = observed_reflectance[valid_indices]

    println("🔹 Valid Wavelengths: ", valid_wavelengths)
    println("🔹 Valid Reflectance Values: ", valid_reflectance)

    if isempty(valid_wavelengths) || isempty(valid_reflectance)
        error("❌ No valid reflectance data for pixel ($i, $j).")
    end

    user_continue("Pixel selection completed")

    return valid_wavelengths, valid_reflectance
end

# Initialize ensemble
function initialize_ensemble(num_ensemble::Int, prior_means::Vector{Float64}, prior_stds::Vector{Float64})
    println("\n🚀 Initializing ensemble...")

    rng = MersenneTwister(42)
    ensemble = hcat([randn(num_ensemble) .* prior_stds[i] .+ prior_means[i] for i in 1:length(prior_means)]...)  

    println("✅ Ensemble initialized: Shape -> ", size(ensemble))
    println("🔹 First 5 ensemble members: ", ensemble[:, 1:5])

    user_continue("Ensemble initialized")

    return ensemble
end

@everywhere begin
    function target_curve(ref_x::Vector{Float64}, params::Dict{String, Float64})
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
            if (_min_wl <= ref_x[i] <= _max_wl) && !(1790 <= ref_x[i] <= 1920) && !(1345 <= ref_x[i] <= 1415)
                _tar_ys[i] = read_spectrum(CONFIG.SPECTRA.Λ, SHIFT.canopy.sensor_geometry.auxil.reflectance, ref_x[i])
            else
                _tar_ys[i] = NaN
            end
        end

        valid_indices = .!isnan.(_tar_ys)
        return _tar_ys[valid_indices]
    end
end


# Simulate reflectance
function simulate_reflectance(wavelengths, params_dict)
    println("\n🔬 Running reflectance simulation...")

    simulated_reflectance = target_curve(wavelengths, params_dict)

    println("🔹 Simulated Reflectance: ", simulated_reflectance)

    if isempty(simulated_reflectance) || all(isnan.(simulated_reflectance))
        error("❌ Simulated reflectance contains only NaNs!")
    end

    user_continue("Simulated reflectance generated")

    return simulated_reflectance
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

# Fit traits using EKI
function fit_shift_traits_eki(valid_wavelengths, valid_reflectance)
    println("\n🚀 Running Ensemble Kalman Inversion...")

    num_ensemble = 80  
    prior_means = [40.0, 3.0, 0.012, 7.5, 0.010, 0.0095, 17.0, 0.5]  
    prior_stds  = [30.0, 1.5, 0.010, 5.0, 0.005, 0.0085, 2.0, 0.4]   

    ensemble = initialize_ensemble(num_ensemble, prior_means, prior_stds)

    observation_noise = 0.005
    Γ = Diagonal(observation_noise^2 * ones(length(valid_wavelengths)))
    
    prior_distributions = [
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("cab", prior_means[1], prior_stds[1], 0.01, 80),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("lai", prior_means[2], prior_stds[2], 0.01, 7),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("lma", prior_means[3], prior_stds[3], 1e-6, 0.05),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("lwc", prior_means[4], prior_stds[4], 0.1, 20),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("cbc", prior_means[5], prior_stds[5], 0.0005, 0.035),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("pro", prior_means[6], prior_stds[6], 1e-6, 0.020),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("sc", prior_means[7], prior_stds[7], 1, 20),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("tsm", prior_means[8], prior_stds[8], 0.01, 0.99)
        ]
    prior_combined = EnsembleKalmanProcesses.ParameterDistributions.combine_distributions(prior_distributions)

    eki = EnsembleKalmanProcess(ensemble', valid_reflectance, Γ, Inversion(); rng=MersenneTwister(42))

    for step in 1:2
        #params_i = EnsembleKalmanProcesses.get_ϕ_final(eki)
        params_i = EnsembleKalmanProcesses.get_ϕ_final(prior_combined, eki)

        println("🔎 EKI Iteration $step: ", params_i[:, 1:5])

        G_ensemble = hcat([target_curve(valid_wavelengths, solver_dict_func(params_i[:, j])) for j in 1:num_ensemble]...)

        if size(G_ensemble, 1) ≠ size(valid_reflectance, 1)
            println("⚠ Warning: Dimension mismatch at:")
            println("   - Observed reflectance size: $(size(valid_reflectance))")
            println("   - Simulated reflectance size: $(size(G_ensemble))")
            println("   - Observed reflectance sample: ", valid_reflectance[1:min(5, length(valid_reflectance))])
            println("   - Simulated reflectance sample: ", G_ensemble[1:min(5, size(G_ensemble, 1)), :])
            println("⚠ Skipping update...")
            continue
        end

        if size(G_ensemble, 1) ≠ size(valid_reflectance, 1)
            println("⚠ Warning: Dimension mismatch. Skipping update.")
            continue
        end

        EnsembleKalmanProcesses.update_ensemble!(eki, G_ensemble)
    end

    optimized_params = EnsembleKalmanProcesses.get_ϕ(prior_combined, eki)
    println("✅ EKI Optimization Complete: ", optimized_params)

    user_continue("EKI completed")

    return optimized_params
end

# Main function
function fit_shift_traits_eki!(datafile::String)
    wavelengths, reflectance, lat_values, lon_values = read_validate_data(datafile)

    valid_wavelengths, valid_reflectance = select_pixel(reflectance, wavelengths, 50, 50)

    params_dict = Dict("cab" => 40.0, "lai" => 5.0, "lma" => 0.012, "lwc" => 5.0, 
                       "cbc" => 0.01, "pro" => 0.005, "sc"  => 10., "tsm" => 0.5)

    simulated_reflectance = simulate_reflectance(valid_wavelengths, params_dict)
    optimized_params = fit_shift_traits_eki(valid_wavelengths, valid_reflectance)

    println("✅ Final Optimized Parameters: ", optimized_params)

     # Convert from Vector of Matrices to a single Matrix
    optimized_matrix = hcat(optimized_params...)  

    # Compute mean and standard deviation along each parameter (row-wise)
    optimized_mean = mean(optimized_matrix, dims=2)
    optimized_std = std(optimized_matrix, dims=2)

    # Define parameter names
    parameter_names = ["cab", "lai", "lma", "lwc", "cbc", "pro", "sc", "tsm"]

    # Print results
    println("\n✅ Optimized parameters (mean ± std):")
    for i in 1:length(parameter_names)
        if parameter_names[i] == "sc"
            println("$(parameter_names[i]) = $(round(Int, optimized_mean[i])) ± $(round(Int, optimized_std[i]))")  # Integer rounding for sc
        else
            println("$(parameter_names[i]) = $(round(optimized_mean[i]; digits=4)) ± $(round(optimized_std[i]; digits=4))")
        end
    end


    println("✅ Debugging complete! Results saved.")
end

# Run the function
datafile = "merged_output_subset.nc"
fit_shift_traits_eki!(datafile)
