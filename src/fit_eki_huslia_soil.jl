using Distributed
using NetcdfIO: append_nc!, create_nc!, read_nc
using ProgressMeter: @showprogress
using Base.GC
using Printf
using Random
using Statistics
using LinearAlgebra  # For identity matrix and matrix operations

# Ensure multiple processes are available
if length(workers()) == 1
    addprocs(1; exeflags = "--project")
end

# Load EnsembleKalmanProcesses on all worker processes
@everywhere begin
    using EnsembleKalmanProcesses
    using Emerald.EmeraldLand.Namespace: BulkSPAC, SPACConfiguration
    using Emerald.EmeraldLand.SPAC: initialize_spac!, prescribe_traits!, read_spectrum, soil_plant_air_continuum!
    using Emerald.EmeraldMath.Stats: rmse
    using Random
    using LinearAlgebra  # Import LinearAlgebra for identity matrix

    # Global configuration
    FT = Float64
    CONFIG = SPACConfiguration(FT)
    CONFIG.ENABLE_SIF = false
    SHIFT = BulkSPAC(CONFIG)
    initialize_spac!(CONFIG, SHIFT)
    SHIFT_BAK = deepcopy(SHIFT)

    # Function to initialize ensemble with configurable parameters (8 parameters)
    function initialize_ensemble(num_ensemble::Int, prior_means::Vector{FT}, prior_stds::Vector{FT}, min_vals::Vector{FT}, max_vals::Vector{FT}, perturbation::FT)
        rng = MersenneTwister(42)
        ensemble = hcat([randn(num_ensemble) .* prior_stds[i] .+ prior_means[i] for i in 1:length(prior_means)]...)  

        # Rescale the ensemble values to be within the specified ranges for each parameter
        for i in 1:length(prior_means)
            ensemble[i, :] .= min_vals[i] .+ (max_vals[i] - min_vals[i]) .* (ensemble[i, :] .- minimum(ensemble[i, :])) ./ (maximum(ensemble[i, :]) - minimum(ensemble[i, :]))
        end

        # Add large perturbations to ensure diversity in the ensemble
        ensemble .+= perturbation .* randn(num_ensemble, length(prior_means)) 

        return ensemble
    end

    # Target function for reflectance simulation (now includes soil color and moisture)
    function target_curve(ref_x::Vector{FT}, params::Dict{String, FT}) where {FT}
        println("Parameters passed to target_curve: ")
        for (key, value) in params
            println("  $key = $value")
        end

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
                _tar_ys[i] = FT(NaN)
            end
        end
        
        return _tar_ys
    end

    # Solver function to map parameters to dictionary
    function solver_dict_func(vals::Vector{FT})
        return Dict(
            "cab" => vals[1],
            "lai" => vals[2],
            "lma" => vals[3],
            "lwc" => vals[4],
            "cbc" => vals[5],
            "pro" => vals[6],
            "sc"  => FT(Int(round(clamp(vals[7], 1, 20)))),
            #"sc"  => vals[7],
            "tsm" => vals[8]
        )
    end

    # Main fitting function with EKI
    function fit_shift_traits_eki(ref_xy::Tuple)
        if all(isnan, ref_xy[2][1:10])
            return fill(NaN, 8), fill(NaN, 8)
        end

        function misfit_function(params::AbstractMatrix{FT})
            param_dicts = [solver_dict_func(params[:, j]) for j in 1:size(params, 2)]
            simulated_reflectance = hcat([target_curve(ref_xy[1], param_dict) for param_dict in param_dicts]...)
            return simulated_reflectance .- ref_xy[2]
        end

        num_ensemble = 80  
        prior_means = [40.0, 3.0, 0.012, 7.5, 0.010, 0.0095, 17.0, 0.5]  
        prior_stds  = [30.0, 1.5, 0.010, 5.0, 0.005, 0.0085, 2.0, 0.4]   

        ensemble = initialize_ensemble(num_ensemble, prior_means, prior_stds, [0.01, 0.01, 1e-6, 0.1, 0.0005, 1e-6, 1, 0.01], [80, 7, 0.05, 20, 0.035, 0.020, 20, 0.99], 0.1)

        observation_noise = 0.005
        Γ = Diagonal(observation_noise^2 * ones(length(ref_xy[1])))  

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

        eki = EnsembleKalmanProcess(ensemble', ref_xy[2], Γ, Inversion(); rng=MersenneTwister(42))

        for i in 1:20
            params_i = EnsembleKalmanProcesses.get_ϕ_final(prior_combined, eki)  
            G_ensemble = hcat([target_curve(ref_xy[1], solver_dict_func(params_i[:, j])) for j in 1:num_ensemble]...)
            EnsembleKalmanProcesses.update_ensemble!(eki, G_ensemble)
        end

        optimized_params = EnsembleKalmanProcesses.get_ϕ(prior_combined, eki)  
        return optimized_params
    end
end

# Main function to fit traits for a specific pixel
function fit_shift_traits_eki!(datafile::String)
    println("Reading data from file: ", datafile)
    wavelengths = read_nc(FT, datafile, "wavelength")
    reflectance = read_nc(FT, datafile, "Reflectance")
    lat_values  = read_nc(FT, datafile, "lat")  
    lon_values  = read_nc(FT, datafile, "lon")  

    # Select a single pixel for debugging
    i, j = 330, 330  # Pixel index for debugging
    observed_reflectance = reflectance[i, j, :]
    println("Observed reflectance for pixel ($i, $j): ", observed_reflectance)

    # Remove NaNs from the observed reflectance
    valid_indices = .!isnan.(observed_reflectance)
    valid_wavelengths = wavelengths[valid_indices]
    valid_reflectance = observed_reflectance[valid_indices]
    println("Valid wavelengths after NaN filtering: ", valid_wavelengths)
    println("Valid reflectance values after NaN filtering: ", valid_reflectance)

    # Ensure valid reflectance values are between 0 and 1
    valid_reflectance = clamp.(valid_reflectance, 0.0, 1.0)
    println("Clamped reflectance values: ", valid_reflectance)

    # Generate simulated reflectance
    # Create the dictionary of parameters to pass to target_curve
    params_dict = Dict(
        "cab" => 40.0,  # Example value for 'cab'
        "lai" => 5.0,   # Example value for 'lai'
        "lma" => 0.012,  # Example value for 'lma'
        "lwc" => 5.0,    # Example value for 'lwc'
        "cbc" => 0.01,   # Example value for 'cbc'
        "pro" => 0.005,   # Example value for 'pro'
        "sc"  => 10.,     # Example value for 'sc'
        "tsm" => 0.5     # Example value for 'tsm'
                    )
    simulated_reflectance = target_curve(valid_wavelengths, params_dict)  # Example with "LAI" = 3.0
    println("Simulated reflectance for inital values: ", simulated_reflectance)

    # Filter both observed and simulated reflectance by removing NaNs
    valid_simulated_indices = .!isnan.(simulated_reflectance)
    valid_reflectance = valid_reflectance[valid_simulated_indices]
    valid_simulated_reflectance = simulated_reflectance[valid_simulated_indices]
    valid_wavelengths = valid_wavelengths[valid_simulated_indices]
    println("Filtered wavelengths: ", valid_wavelengths)
    println("Filtered observed reflectance: ", valid_reflectance)
    println("Filtered simulated reflectance: ", valid_simulated_reflectance)

    # Call the fitting function
    result = fit_shift_traits_eki((valid_wavelengths, valid_reflectance))

    # Output the result
    #println("Optimized parameters: ", result)

    # Extract the array from Any
    parameters = hcat(result...)  # Convert list of arrays to a matrix

    # Compute mean and standard deviation across columns
    optimized_mean = mean(parameters, dims=2)
    optimized_std = std(parameters, dims=2)

    # Define parameter names
    parameter_names = ["cab", "lai", "lma", "lwc", "cbc", "pro", "sc", "tsm"]

    # Print the results in the requested format
    println("Optimized parameters (mean ± std):")
    for i in 1:length(parameter_names)
        if parameter_names[i] == "sc"
            println("$(parameter_names[i]) = $(round(Int, optimized_mean[i])) ± $(round(Int, optimized_std[i]))")  # Force integer output for sc
        else
            println("$(parameter_names[i]) = $(round(optimized_mean[i], digits=4)) ± $(round(optimized_std[i], digits=4))")
        end
    end

end

# Run the function
datafile = "merged_output.nc"  # Replace with the actual data file
fit_shift_traits_eki!(datafile)
