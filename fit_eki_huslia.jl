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

    # Function to initialize ensemble with configurable parameters (6 parameters)
    function initialize_ensemble(num_ensemble::Int, prior_means::Vector{FT}, prior_stds::Vector{FT}, min_vals::Vector{FT}, max_vals::Vector{FT}, perturbation::FT)
        rng = MersenneTwister(42)
        ensemble = hcat([randn(num_ensemble) .* prior_stds[i] .+ prior_means[i] for i in 1:length(prior_means)]...)  # Gaussian prior for all variables

        # Rescale the ensemble values to be within the specified ranges for each parameter
        for i in 1:length(prior_means)
            ensemble[i, :] .= min_vals[i] .+ (max_vals[i] - min_vals[i]) .* (ensemble[i, :] .- minimum(ensemble[i, :])) ./ (maximum(ensemble[i, :]) - minimum(ensemble[i, :]))
        end
        
        # Add large perturbations to ensure diversity in the ensemble
        ensemble .+= perturbation .* randn(num_ensemble, length(prior_means)) 

        return ensemble
    end

    # Updated target function for reflectance simulation (for all 6 parameters)
    function target_curve(ref_x::Vector{FT}, params::Dict{String, FT}) where {FT}
        println("Parameters passed to target_curve: ")
        for (key, value) in params
            println("  $key = $value")
        end

        SHIFT = deepcopy(SHIFT_BAK)
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
                _tar_ys[i] = FT(NaN)  # Set out-of-bounds wavelengths to NaN
            end
        end
        return _tar_ys
    end

    # Solver function to map the parameter vector to a dictionary of trait values
    function solver_dict_func(vals::Vector{FT})
        return Dict(
            "cab" => vals[1],
            "lai" => vals[2],
            "lma" => vals[3],
            "lwc" => vals[4],
            "cbc" => vals[5],
            "pro" => vals[6]
        )
    end

    # Main fitting function with EKI (estimate all 6 traits)
    function fit_shift_traits_eki(ref_xy::Tuple)
        if all(isnan, ref_xy[2][1:10])
            return fill(NaN, 6), fill(NaN, 6)
        end

        function misfit_function(params::AbstractMatrix{FT})
            param_dicts = [solver_dict_func(params[:, j]) for j in 1:size(params, 2)]
            simulated_reflectance = hcat([target_curve(ref_xy[1], param_dict) for param_dict in param_dicts]...)
            
            return simulated_reflectance .- ref_xy[2]
        end

        # Ensemble initialization with variation for leaf traits
        num_ensemble = 60  # Minimum ensemble size
        #["cab", "lai", "lma", "lwc", "cbc", "pro"]
        prior_means = [40.0, 5.0, 0.012, 7.5, 0.010, 0.0095]  # Means for each parameter
        prior_stds = [30.0, 4.0, 0.010, 5.0, 0.005, 0.0085]  # Std devs for each parameter

        # Initialize ensemble using prior means and standard deviations
        ensemble = hcat([randn(num_ensemble) .* prior_stds[i] .+ prior_means[i] for i in 1:length(prior_means)]...)  # Correct shape (num_ensemble, num_params)
        println("Ensemble initialized with shape: ", size(ensemble))  # Ensure it prints (num_ensemble, num_params)

        # Observation noise covariance (diagonal matrix)
        Γ = Diagonal(0.01^2 * ones(length(ref_xy[1])))  # Diagonal covariance matrix

        # Set up prior distributions for parameters
        prior_distributions = [
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("cab", prior_means[1], prior_stds[1], 0.01, 80),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("lai", prior_means[2], prior_stds[2], 0.01, 10),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("lma", prior_means[3], prior_stds[3], 1e-6, 0.05),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("lwc", prior_means[4], prior_stds[4], 0.1, 20),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("cbc", prior_means[5], prior_stds[5], 0.0005, 0.035),
            EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("pro", prior_means[6], prior_stds[6], 1e-6, 0.020)
        ]
        prior_combined = EnsembleKalmanProcesses.ParameterDistributions.combine_distributions(prior_distributions)

        # Set up EKI process
        eki = EnsembleKalmanProcess(
            ensemble',  # Ensure the ensemble is transposed to (num_ensemble, num_params)
            ref_xy[2],  # Observed reflectance
            Γ, 
            Inversion();
            rng=MersenneTwister(42),
            scheduler=DataMisfitController(on_terminate="continue")  # Prevent early stopping
        )

        # Run the EKI updates
        num_iterations = 20  # Increase iterations for more stable results
        for i in 1:num_iterations
            params_i = EnsembleKalmanProcesses.get_ϕ_final(prior_combined, eki)  # Get current ensemble parameters
            G_ensemble = hcat([target_curve(ref_xy[1], solver_dict_func(params_i[:, j])) for j in 1:num_ensemble]...)
            
            # Check for forward map clashes (exact equality of simulations)
            if length(unique(G_ensemble)) < num_ensemble
                println("Warning: Forward map evaluations are too similar, which may cause numerical instability.")
            end

            # Update ensemble with new reflectance simulations
            EnsembleKalmanProcesses.update_ensemble!(eki, G_ensemble)
        end

        # Extract optimized parameters and their uncertainty
        optimized_params = EnsembleKalmanProcesses.get_ϕ(prior_combined, eki)  # Retrieve the ensemble parameters

        return optimized_params
    end
end  # End @everywhere block

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
        "pro" => 0.005   # Example value for 'pro'
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
    parameter_names = ["cab", "lai", "lma", "lwc", "cbc", "pro"]

    # Print the results in the requested format
    println("Optimized parameters (mean ± std):")
    for i in 1:length(parameter_names)
        println("$(parameter_names[i]) = $(round(optimized_mean[i], digits=4)) ± $(round(optimized_std[i], digits=4))")
    end

end

# Run the function
datafile = "merged_output.nc"  # Replace with the actual data file
fit_shift_traits_eki!(datafile)
