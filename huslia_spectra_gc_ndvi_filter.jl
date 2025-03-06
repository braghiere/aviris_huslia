using NetcdfIO: append_nc!, create_nc!, read_nc
using Distributed: @everywhere, addprocs, pmap, workers, rmprocs
using ProgressMeter: @showprogress
using Base.GC
using Printf
using Statistics

# Ensure multiple processes are available for parallel computation
# ---- Optimize Worker Management ----
if length(workers()) > 1
    rmprocs(workers())  # Ensure clean start
end
if length(workers()) == 1
    addprocs(48; exeflags = "--project")
end

# Load the target function for spectral fitting
@everywhere include("target_function_v3.jl")

"""
    compute_ndvi(reflectance, wavelengths)

Computes NDVI using a small band for Red (660-680 nm) and NIR (840-860 nm).

Returns:
- `ndvi` value for the pixel.
"""
function compute_ndvi(reflectance, wavelengths)
    idx_red = findfirst(x -> 660 ≤ x ≤ 680, wavelengths)
    idx_nir = findfirst(x -> 840 ≤ x ≤ 860, wavelengths)

    if idx_red === nothing || idx_nir === nothing
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
    # Read data from NetCDF file
    wavelengths = read_nc(Float64, datafile, "wavelength")
    reflectance = read_nc(Float64, datafile, "Reflectance")
    lat_values  = read_nc(Float64, datafile, "lat")  
    lon_values  = read_nc(Float64, datafile, "lon")  

    reflectance = clamp.(reflectance, 0.0, 1.0)

    #dimensions:
	#lon = 660 ;
	#lat = 5890 ;

    px_range_i = 1:660
    px_range_j = 1:5890

    # Map global indices to local indices
    px_i_to_local = Dict(px => i for (i, px) in enumerate(px_range_i))
    px_j_to_local = Dict(px => j for (j, px) in enumerate(px_range_j))

    params = []
    valid_indices = []  # Store valid local indices for matrix assignment

    for i in px_range_i, j in px_range_j
        ndvi = compute_ndvi(reflectance[i, j, :], wavelengths)

        if ndvi > 0.2  # Only process vegetation pixels
            push!(params, (deepcopy(wavelengths), reflectance[i, j, :], i, j))
            push!(valid_indices, (px_i_to_local[i], px_j_to_local[j]))  # Store local indices
        end
    end

    fittings = @showprogress pmap(fit_shift_traits, params)

    # Define trait names and initialize matrices dynamically
    trait_names = ["chl", "lai", "lma", "lwc", "cbc", "pro"]
    lat_size = length(px_range_i)
    lon_size = length(px_range_j)
    
    trait_matrices = Dict(trait => fill(NaN, lat_size, lon_size) for trait in trait_names)

    for (idx, (i_local, j_local)) in enumerate(valid_indices)
        for (trait_idx, trait) in enumerate(trait_names)
            trait_matrices[trait][i_local, j_local] = fittings[idx][trait_idx]
        end
        # Handle lma separately (lma = cbc + pro)
        trait_matrices["lma"][i_local, j_local] = fittings[idx][5] + fittings[idx][6]
    end

    # Save the results to a NetCDF file
    create_nc!(ncresult, ["lon", "lat"], [lon_size, lat_size])  
    append_nc!(ncresult, "lat", lat_values[px_range_i], Dict("latitude" => "latitude"), ["lat"])
    append_nc!(ncresult, "lon", lon_values[px_range_j], Dict("longitude" => "longitude"), ["lon"])

    for trait in trait_names
        append_nc!(ncresult, trait, trait_matrices[trait], Dict(trait => trait), ["lon", "lat"])
    end

    GC.gc()

    return fittings
end

# Run the function
#datafile = "data/merged_output_subset.nc"
#outputfile = "data/test_output_rmse_ndvi.nc"
datafile = "data/merged_output.nc"
outputfile = "data/output_rmse_ndvi.nc"
fit_shift_traits!(datafile, outputfile)
