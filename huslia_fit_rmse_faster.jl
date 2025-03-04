using NetcdfIO: append_nc!, create_nc!, read_nc
using Distributed: @everywhere, addprocs, pmap, workers, rmprocs
using ProgressMeter: @showprogress
using Base.GC
using Printf
using Statistics

# ---- Optimize Worker Management ----
if length(workers()) > 1
    rmprocs(workers())  # Ensure clean start
end
addprocs(64; exeflags = "--project")  # Reduce to 16 for better efficiency

# Load target function across all workers
@everywhere include("target_function_v3.jl")

"""
    compute_ndvi(reflectance, wavelengths)

Computes NDVI using a small band for Red (660-680 nm) and NIR (840-860 nm).
"""
function compute_ndvi(reflectance, wavelengths)
    idx_red = findall(x -> 660 ≤ x ≤ 680, wavelengths)
    idx_nir = findall(x -> 840 ≤ x ≤ 860, wavelengths)

    if isempty(idx_red) || isempty(idx_nir)
        return NaN
    end

    red = mean(reflectance[idx_red])  
    nir = mean(reflectance[idx_nir])  

    return (nir - red) / (nir + red + 1e-6)
end

"""
    fit_shift_traits!(datafile::String, ncresult::String)

Fits spectral shift traits from a NetCDF file, filtering out non-vegetation pixels.
"""
function fit_shift_traits!(datafile::String, ncresult::String)
    # ---- Read NetCDF Data ----
    wavelengths = read_nc(Float64, datafile, "wavelength")
    reflectance = read_nc(Float64, datafile, "Reflectance")
    lat_values  = read_nc(Float64, datafile, "lat")  
    lon_values  = read_nc(Float64, datafile, "lon")  

    reflectance .= clamp.(reflectance, 0.0, 1.0)  # In-place modification

    # ---- Define Grid ----
    px_range_i = 1:min(101, size(reflectance, 1))  # Ensure safe indexing
    px_range_j = 1:min(101, size(reflectance, 2))

    # ---- Precompute Index Mapping ----
    px_i_to_local = Dict(px => i for (i, px) in enumerate(px_range_i))
    px_j_to_local = Dict(px => j for (j, px) in enumerate(px_range_j))

    # ---- Collect Valid Pixels ----
    params = Tuple[]
    valid_indices = Tuple[]

    @inbounds for i in px_range_i, j in px_range_j  # Speed up loops
        ndvi = compute_ndvi(view(reflectance, i, j, :), wavelengths)

        if ndvi > 0.2  # Only process vegetation pixels
            push!(params, (wavelengths, reflectance[i, j, :], i, j))
            push!(valid_indices, (px_i_to_local[i], px_j_to_local[j]))
        end
    end

    # ---- Parallel Processing ----
    fittings = @showprogress pmap(fit_shift_traits, params; batch_size=50)  # Reduce batch size for better balancing

    # ---- Initialize Matrices ----
    trait_names = ["chl", "lai", "lma", "lwc", "cbc", "pro"]
    lat_size, lon_size = length(px_range_i), length(px_range_j)

    trait_matrices = Dict(trait => fill(NaN, lat_size, lon_size) for trait in trait_names)

    # ---- Assign Computed Values ----
    @inbounds for (idx, (i_local, j_local)) in enumerate(valid_indices)
        if idx <= length(fittings) && length(fittings[idx]) == 6  # Ensure no out-of-bounds
            for (trait_idx, trait) in enumerate(trait_names)
                trait_matrices[trait][i_local, j_local] = fittings[idx][trait_idx]
            end
            trait_matrices["lma"][i_local, j_local] = fittings[idx][5] + fittings[idx][6]
        else
            @warn "Skipping index ($i_local, $j_local) due to incomplete result"
        end
    end

    # ---- Save to NetCDF ----
    create_nc!(ncresult, ["lon", "lat"], [lon_size, lat_size])  
    append_nc!(ncresult, "lat", lat_values[px_range_i], Dict("latitude" => "latitude"), ["lat"])
    append_nc!(ncresult, "lon", lon_values[px_range_j], Dict("longitude" => "longitude"), ["lon"])

    for trait in trait_names
        append_nc!(ncresult, trait, trait_matrices[trait], Dict(trait => trait), ["lon", "lat"])
    end

    GC.gc(false)  # Force garbage collection to free memory
    return fittings
end

# ---- Run the Optimized Function ----
datafile = "data/merged_output_subset.nc"
outputfile = "data/test_output_rmse_ndvi.nc"
fit_shift_traits!(datafile, outputfile)
