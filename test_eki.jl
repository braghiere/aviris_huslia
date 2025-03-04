using Pkg
Pkg.activate(@__DIR__)  # Ensure correct environment
Pkg.instantiate()

using LinearAlgebra  # Import LinearAlgebra for identity matrix
using EnsembleKalmanProcesses  # Import EKI package
using Random  # For reproducibility

println("Running EKI Test...")

function test_eki()
    println("Running EKI Test...")

    # Define the observation model (maps parameters to observations)
    function G(params::AbstractVector{Float64})
        return 2.0 .* params .+ 1.0  # A simple linear transformation
    end

    # Define true parameters and generate synthetic observations
    true_params = [3.0, 1.0, 2.0]
    observation_noise = 0.1
    y = G(true_params) .+ observation_noise * randn(3)  # Add Gaussian noise

    # Define prior distributions (Gaussian)
    prior_u1 = EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("param1", 0.0, 5.0, -Inf, Inf)
    prior_u2 = EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("param2", 0.0, 5.0, -Inf, Inf)
    prior_u3 = EnsembleKalmanProcesses.ParameterDistributions.constrained_gaussian("param3", 0.0, 5.0, -Inf, Inf)
    prior = EnsembleKalmanProcesses.ParameterDistributions.combine_distributions([prior_u1, prior_u2, prior_u3])

    # Increase the ensemble size (fix warning)
    num_params = 3
    num_ensemble = 30  # ✅ Increased to meet best practices
    rng = MersenneTwister(42)  # For reproducibility
    ensemble = EnsembleKalmanProcesses.construct_initial_ensemble(rng, prior, num_ensemble)

    # Define observation noise covariance (diagonal matrix)
    Γ = observation_noise^2 * I(num_params)

    # Create an Ensemble Kalman Process with a modified scheduler
    eki = EnsembleKalmanProcess(ensemble, y, Γ, Inversion();
        rng=rng,
        scheduler=DataMisfitController(on_terminate="continue")  # ✅ Prevent early stopping
    )

    # Run the EKI updates
    num_iterations = 10  # ✅ Increased for better convergence
    for i in 1:num_iterations
        params_i = EnsembleKalmanProcesses.get_ϕ_final(prior, eki)  # Get current ensemble parameters
        G_ensemble = hcat([G(params_i[:, j]) for j in 1:num_ensemble]...)  # Apply the model to the ensemble
        EnsembleKalmanProcesses.update_ensemble!(eki, G_ensemble)  # Update ensemble
    end

    # Get optimized parameters
    optimized_params = EnsembleKalmanProcesses.get_ϕ_mean_final(prior, eki)

    # Print results
    println("Optimized Parameters: ", optimized_params)

    return optimized_params
end

# Run the test
test_eki()
