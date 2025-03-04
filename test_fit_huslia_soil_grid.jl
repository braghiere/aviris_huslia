using Distributed
using NetcdfIO: append_nc!, create_nc!, read_nc
@everywhere using ProgressMeter
using Base.GC
using Printf
using Random
using Statistics
using LinearAlgebra  # For identity matrix and matrix operations

# Ensure multiple processes are available
if length(workers()) == 1
    addprocs(1; exeflags = "--project")
end

# Load necessary modules on all workers
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

    function target_curve(ref_x::Vector{FT}, params::Dict{String, FT}) where {FT}
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

        if "sc" in keys(params)
            SHIFT.soil_bulk.trait.color = Int(round(params["sc"]))
        end
        if "tsm" in keys(params)
            SHIFT.soils[1].state.θ = params["tsm"]
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
                _tar_ys[i] = FT(NaN)
            end
        end
        return _tar_ys
    end

    function fit_shift_traits_eki_grid!(datafile::String, outputfile::String)
        println("Reading data from file: ", datafile)
        wavelengths = read_nc(Float64, datafile, "wavelength")
        reflectance = read_nc(Float64, datafile, "Reflectance")
        lat_values  = read_nc(Float64, datafile, "lat")  
        lon_values  = read_nc(Float64, datafile, "lon")  
    
        # Define grid range (330:340, 330:340) 
        i_range = 330:340
        j_range = 330:340
    
        # Prepare output NetCDF file
        create_nc!(outputfile, Dict(
            "lat" => lat_values[i_range],
            "lon" => lon_values[j_range],
            "parameters" => Array{Float64}(undef, length(i_range), length(j_range), 8),  # 8 parameters
            "stddev" => Array{Float64}(undef, length(i_range), length(j_range), 8)  # Standard deviations
        ))
    
        @showprogress for i in i_range
            for j in j_range
                observed_reflectance = reflectance[i, j, :]
                
                # Remove NaNs
                valid_indices = .!isnan.(observed_reflectance)
                if sum(valid_indices) == 0
                    continue  # Skip if all values are NaN
                end
                
                valid_wavelengths = wavelengths[valid_indices]
                valid_reflectance = observed_reflectance[valid_indices]
    
                # Generate simulated reflectance
                params_dict = Dict(
                    "cab" => 40.0,
                    "lai" => 5.0,
                    "lma" => 0.012,
                    "lwc" => 5.0,
                    "cbc" => 0.01,
                    "pro" => 0.005,
                    "sc"  => 10.0,
                    "tsm" => 0.5
                )
                simulated_reflectance = target_curve(valid_wavelengths, params_dict)
                
                # Filter NaNs in both observed and simulated reflectance
                valid_simulated_indices = .!isnan.(simulated_reflectance)
                valid_reflectance = valid_reflectance[valid_simulated_indices]
                valid_simulated_reflectance = simulated_reflectance[valid_simulated_indices]
                valid_wavelengths = valid_wavelengths[valid_simulated_indices]
                
                if length(valid_reflectance) == 0
                    continue
                end
                
                # Fit traits using EKI
                result = fit_shift_traits_eki((valid_wavelengths, valid_reflectance))
                parameters = hcat(result...)
                
                # Compute mean and standard deviation
                optimized_mean = mean(parameters, dims=2)
                optimized_std = std(parameters, dims=2)
    
                # Save to NetCDF file
                append_nc!(outputfile, "parameters", optimized_mean, indices=(i - i_range[1] + 1, j - j_range[1] + 1, :))
                append_nc!(outputfile, "stddev", optimized_std, indices=(i - i_range[1] + 1, j - j_range[1] + 1, :))
            end
        end
        println("Processing complete. Results saved to ", outputfile)
    end
end

# Run the function
datafile = "merged_output.nc"
outputfile = "optimized_parameters.nc"
fit_shift_traits_eki_grid!(datafile, outputfile)
