using Distributed
using NetcdfIO: append_nc!, create_nc!, read_nc
using ProgressMeter: @showprogress
using Base.GC
using Random
using Statistics
using LinearAlgebra  

# 1) Ensure multiple processes
rmprocs(workers())
if length(workers()) == 1
    addprocs(4; exeflags="--project")
end

# 2) Load needed packages *everywhere*
@everywhere begin
    using EnsembleKalmanProcesses
    using Emerald.EmeraldLand.Namespace: BulkSPAC, SPACConfiguration
    using Emerald.EmeraldLand.SPAC: initialize_spac!, prescribe_traits!, read_spectrum, soil_plant_air_continuum!
    using Emerald.EmeraldMath.Stats: rmse
    using ProgressMeter: @showprogress
    using LinearAlgebra
    using Statistics
    using Random

    const FT = Float64

    # Build SHIFT_BAK
    CONFIG = SPACConfiguration(FT)
    CONFIG.ENABLE_SIF = false
    SHIFT = BulkSPAC(CONFIG)
    initialize_spac!(CONFIG, SHIFT)
    SHIFT_BAK = deepcopy(SHIFT)

    # -- Initialize Ensemble
    @everywhere function initialize_ensemble(num_ensemble::Int,
                                             prior_means::Vector{FT},
                                             prior_stds::Vector{FT},
                                             min_vals::Vector{FT},
                                             max_vals::Vector{FT},
                                             perturbation::FT)
        rng = MersenneTwister(42)
        # shape = (num_params, num_ensemble)
        ensemble = transpose(hcat([
            randn(num_ensemble) .* prior_stds[i] .+ prior_means[i]
            for i in 1:length(prior_means)
        ]...))

        # Rescale each parameter
        for p in 1:size(ensemble, 1)
            mn, mx = min_vals[p], max_vals[p]
            range_diff = maximum(ensemble[p, :]) - minimum(ensemble[p, :])
            if range_diff > 1e-10
                ensemble[p, :] .= mn .+ (mx-mn).*(ensemble[p,:] .- minimum(ensemble[p,:])) ./ range_diff
            else
                ensemble[p, :] .= mean([mn, mx])
            end
        end

        # Add small perturbations
        ensemble .+= perturbation .* randn(size(ensemble,1), size(ensemble,2))
        return ensemble
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
end

# Debugging EKI per-pixel
@everywhere function fit_shift_traits_eki(wls::Vector{Float64}, refl::Vector{Float64}, i::Int, j::Int)
    @info "Pixel ($i,$j): starting EKI..."

    # Check for NaNs
    if all(isnan.(refl))
        @warn "All reflectance values are NaN for pixel ($i,$j). Skipping."
        return fill(FT(NaN), 8), (i,j)
    end

    wls_f = wls[.!isnan.(refl)]
    refl_f = refl[.!isnan.(refl)]

    # Check if there are valid wavelengths
    if length(wls_f) == 0
        @warn "No valid wavelengths after filtering for pixel ($i,$j)."
        return fill(FT(NaN), 8), (i,j)
    end

    # Initialize ensemble
    prior_means = [40.0,3.0,0.012,7.5,0.010,0.0095,17.0,0.5]
    prior_stds  = [30.0,1.5,0.007,5.0, 0.005,0.0025,2.0,0.4]
    ensemble = initialize_ensemble(80, prior_means, prior_stds,
                                   [0.01,0.01,1e-5,0.1,0.0005,1e-5,1,0.01],
                                   [80.0,7.0,0.05,20.0,0.035,0.02,20,0.99], 0.1)

    # Check for NaNs in ensemble
    if any(isnan.(ensemble))
        @warn "NaNs detected in initialized ensemble for pixel ($i,$j)."
        return fill(FT(NaN), 8), (i,j)
    end

    # Observation noise matrix
    Γ = Diagonal((0.005)^2 * ones(length(wls_f)))

    # EKI process
    optimized_params = fill(FT(NaN), 8)
    try
        eki = EnsembleKalmanProcess(ensemble, refl_f, Γ, Inversion(); rng=MersenneTwister(42))

        for step in 1:20
            param_mat = get_ϕ_final(Inversion(), eki)

            if any(isnan.(param_mat))
                @warn "NaN detected in parameter matrix at step $step for pixel ($i,$j)"
                break
            end

            sim_list = []
            for col in 1:size(param_mat,2)
                sim_out = target_curve(wls_f, solver_dict_func(param_mat[:,col]))
                push!(sim_list, sim_out)
            end

            G_ensemble = hcat(sim_list...)
            EnsembleKalmanProcesses.update_ensemble!(eki, G_ensemble; y=refl_f)
        end

        final_ens = calcParamMatrix(eki)
        for p in 1:8
            optimized_params[p] = mean(final_ens[p,:])
        end

        @info "Pixel($i,$j) => Final optimized params: $optimized_params"
    catch e
        @warn "EKI failed at pixel ($i,$j): $e"
    end

    return optimized_params, (i,j)
end

@everywhere function fit_shift_traits_grid!(datafile::String, ncresult::String)
    @info "Reading data from $datafile"

    wls  = read_nc(Float64, datafile, "wavelength")
    refl = read_nc(Float64, datafile, "Reflectance")
    lat  = read_nc(Float64, datafile, "lat")
    lon  = read_nc(Float64, datafile, "lon")

    refl .= clamp.(refl, 0.0,1.0)

    px_range_x = 50:51
    px_range_y = 50:51

    tasks = []
    for i in px_range_x, j in px_range_y
        mask = .!isnan.(refl[i,j,:]) .& (wls .< 2380) .& .!(1345 .<wls.< 1415) .& .!(1790 .<wls.< 1920)
        if any(mask)
            push!(tasks, (wls[mask], refl[i,j,mask], i, j))
        end
    end

    @info "Fitting on $(length(tasks)) tasks"

    # run pmap
    fits = @showprogress pmap(x->fit_shift_traits_eki(x[1], x[2], x[3], x[4]), tasks)

    @info "Saving results to $ncresult"

    pnames = ["cab","lai","lma","lwc","cbc","pro","sc","tsm"]

    # each fits[k] => (8-vector, (i,j))
    param_map = Dict(
        name => [f[1][ip] for f in fits]  # gather the ip-th param from each fit
        for (ip, name) in enumerate(pnames)
    )

    # shape => lat_size= length(px_range_y), lon_size= length(px_range_x)
    lat_size = length(px_range_y)
    lon_size = length(px_range_x)

    # reshape
    shaped = Dict(
        name => reshape(param_map[name], lat_size, lon_size)
        for name in pnames
    )

    # create netcdf
    create_nc!(ncresult, ["lon","lat"], [lon_size, lat_size])
    append_nc!(ncresult, "lat", lat[px_range_y], Dict("latitude"=>"latitude"), ["lat"])
    append_nc!(ncresult, "lon", lon[px_range_x], Dict("longitude"=>"longitude"), ["lon"])

    for name in pnames
        append_nc!(ncresult,
                   name*"_mean",
                   shaped[name],
                   Dict(name*"_mean"=> name*" mean"),
                   ["lon","lat"])
    end

    # print
    for (k,v) in param_map
        println("📊 $k => $v")
    end

    GC.gc()
    @info "Done => results in $ncresult"
end


# run
datafile = "merged_output_subset.nc"
ncresult = "test_fitted_traits_eki.nc"
fit_shift_traits_grid!(datafile, ncresult)
