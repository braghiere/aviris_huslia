using Distributed
rmprocs(workers())
using NetcdfIO: append_nc!, create_nc!, read_nc
using Base.GC
using Printf
using Random
using Statistics
using LinearAlgebra  

# Ensure sufficient workers for parallel execution
if length(workers()) < 4
    addprocs(4 - length(workers()); exeflags = "--project")  
end

# Load dependencies on all workers
@everywhere begin
    using EnsembleKalmanProcesses
    using Emerald.EmeraldLand.Namespace: BulkSPAC, SPACConfiguration
    using Emerald.EmeraldLand.SPAC: initialize_spac!, prescribe_traits!, read_spectrum, soil_plant_air_continuum!
    using Emerald.EmeraldMath.Stats: rmse
    using ProgressMeter
    using Random
    using LinearAlgebra  

    FT = Float64
    CONFIG = SPACConfiguration(FT)
    CONFIG.ENABLE_SIF = false
    SHIFT = BulkSPAC(CONFIG)
    initialize_spac!(CONFIG, SHIFT)
    SHIFT_BAK = deepcopy(SHIFT)

    # Ensure initialize_ensemble is defined
    function initialize_ensemble(num_ensemble, prior_means, prior_stds, lower_bounds, upper_bounds, perturb)
        ens = hcat([randn(length(prior_means)) .* prior_stds .+ prior_means for _ in 1:num_ensemble]...)
        ens = clamp.(ens, lower_bounds, upper_bounds)
        ens .+= randn(size(ens)) * perturb
        return ens
    end

    println("✅ Function initialize_ensemble loaded on worker ", myid())

    # Target function: now filters wavelengths dynamically per (i, j)
    function target_curve(ref_x::Vector{FT}, params::Dict{String, FT}) where {FT}
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

        # Ensure NaNs are removed dynamically
        valid_indices = .!isnan.(_tar_ys)
        return _tar_ys[valid_indices]
    end
end

# Function to fit traits using EKI with dynamic wavelength filtering per (i, j)
@everywhere function fit_shift_traits_eki(wavelengths::Vector{Float64}, reflectance::Vector{Float64}, i::Int, j::Int)
    println("Running fit_shift_traits_eki on worker ", myid(), " for pixel ($i, $j)")

    if isempty(reflectance)
        return fill(NaN, 8), (i, j)
    end

    reflectance = clamp.(reflectance, 0.0, 1.0)

    num_ensemble = 80  
    prior_means = [40.0, 3.0, 0.012, 7.5, 0.010, 0.0095, 17.0, 0.5]  
    prior_stds  = [30.0, 1.5, 0.010, 5.0, 0.005, 0.0085, 2.0, 0.4]   

    try
        ensemble = initialize_ensemble(num_ensemble, prior_means, prior_stds, 
                                       [0.01, 0.01, 1e-6, 0.1, 0.0005, 1e-6, 1, 0.01], 
                                       [80, 7, 0.05, 20, 0.035, 0.020, 20, 0.99], 0.1)
    catch e
        println("⚠️ Error initializing ensemble at ($i, $j): ", e)
        return fill(NaN, 8), (i, j)
    end

    observation_noise = 0.005
    Γ = Diagonal(observation_noise^2 * ones(length(wavelengths)))

    if isempty(wavelengths)
        println("⚠️ Warning: No valid wavelengths at ($i, $j)")
        return fill(NaN, 8), (i, j)
    end

    optimized_params = fill(NaN, 8)  # Default in case of error

    try
        eki = EnsembleKalmanProcess(ensemble', reflectance, Γ, Inversion(); rng=MersenneTwister(42))

        for _ in 1:20
            params_i = EnsembleKalmanProcesses.get_ϕ_final(eki)

            simulated_reflectance_list = [target_curve(wavelengths, solver_dict_func(params_i[:, j])) for j in 1:num_ensemble]

            valid_sim_indices = .!isnan.(simulated_reflectance_list[1])
            filtered_reflectance = reflectance[valid_sim_indices]

            G_ensemble = hcat([sim[valid_sim_indices] for sim in simulated_reflectance_list]...)

            if size(G_ensemble, 1) ≠ size(filtered_reflectance, 1)
                println("⚠ Warning: Dimension mismatch at ($i, $j): G_ensemble size $(size(G_ensemble)) vs observed $(size(filtered_reflectance))")
                continue
            end

            EnsembleKalmanProcesses.update_ensemble!(eki, G_ensemble)
        end

        optimized_params = EnsembleKalmanProcesses.get_ϕ(eki)
    catch e
        println("⚠️ Error during EKI process at ($i, $j): ", e)
    end

    return optimized_params, (i, j)
end

# Main function
function fit_shift_traits_grid!(datafile::String, ncresult::String)
    println("📖 Reading data from file: ", datafile)
    wavelengths = read_nc(Float64, datafile, "wavelength")
    reflectance = read_nc(Float64, datafile, "Reflectance")

    pixel_range = 330:340
    params = []
    for i in pixel_range, j in pixel_range
        valid_indices = .!isnan.(reflectance[i, j, :])
        if any(valid_indices)
            push!(params, (wavelengths[valid_indices], reflectance[i, j, valid_indices], i, j))
        end
    end

    println("🚀 Running parallel fitting on $(length(params)) valid pixels...")
    fittings = pmap(x -> fit_shift_traits_eki(x[1], x[2], x[3], x[4]), params)

    println("💾 Saving results to: ", ncresult)
    create_nc!(ncresult, ["lon", "lat"], (11, 11))
    append_nc!(ncresult, "traits", reshape(hcat([f[1] for f in fittings]...), (8, 11, 11)), Dict("traits" => "Optimized Parameters"), ["lon", "lat"])

    GC.gc()
    println("✅ Fitting complete. Results saved in: ", ncresult)
end

datafile = "merged_output.nc"
ncresult = "test_fitted_traits_eki.nc"
fit_shift_traits_grid!(datafile, ncresult)
