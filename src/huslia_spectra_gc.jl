using NetcdfIO: append_nc!, create_nc!, read_nc
using Distributed: @everywhere, addprocs, pmap, workers
using ProgressMeter: @showprogress
using Base.GC
using Printf

# Ensure multiple processes are available for parallel computation
rmprocs(workers())
if length(workers()) == 1
    addprocs(48; exeflags = "--project")
end

# Load the target function for spectral fitting
@everywhere include("target_function_v3.jl")

"""
    fit_shift_traits!(datafile::String, ncresult::String)

Fits spectral shift traits from a NetCDF file and saves the results in a new NetCDF file.

### Arguments:
- `datafile::String`: Path to the input NetCDF file.
- `ncresult::String`: Path for saving the output NetCDF file.

### Workflow:
1. Reads wavelength, reflectance, latitude, and longitude data.
2. Clamps reflectance values between [0,1].
3. Extracts relevant spectral data for fitting.
4. Uses parallel processing to compute trait fits.
5. Saves the computed traits in a new NetCDF file.
"""
function fit_shift_traits!(datafile::String, ncresult::String)
    # Read data from NetCDF file
    wavelengths = read_nc(FT, datafile, "wavelength")
    reflectance = read_nc(FT, datafile, "Reflectance")
    lat_values  = read_nc(FT, datafile, "lat")  
    lon_values  = read_nc(FT, datafile, "lon")  

    # Ensure reflectance values are within the valid range
    reflectance = clamp.(reflectance, 0.0, 1.0)

    px_range_i = 50:60
    px_range_j = 50:60

    # Prepare parameters for spectral fitting
    params = []
    for i in axes(reflectance,1)[px_range_i], j in axes(reflectance,2)[px_range_j]
        push!(params, (deepcopy(wavelengths), reflectance[i, j, :], i, j))
    end

    # Perform parallel spectral fitting
    fittings = @showprogress pmap(fit_shift_traits, params)

    # Extract individual trait results
    chl_content  = [f[1] for f in fittings]
    lai_content  = [f[2] for f in fittings]
    lma_content  = [f[3] for f in fittings]
    lwc_content  = [f[4] for f in fittings]
    cbc_content  = [f[5] for f in fittings]
    pro_content  = [f[6] for f in fittings]

    # Reshape results to match spatial dimensions
    lat_size = length(axes(reflectance,2)[px_range_i])
    lon_size = length(axes(reflectance,1)[px_range_j])

    mat_chl  = reshape(chl_content, lat_size, lon_size)
    mat_lai  = reshape(lai_content, lat_size, lon_size)
    mat_lma  = reshape(lma_content, lat_size, lon_size)
    mat_lwc  = reshape(lwc_content, lat_size, lon_size)
    mat_cbc  = reshape(cbc_content, lat_size, lon_size)
    mat_pro  = reshape(pro_content, lat_size, lon_size)

    # Save the results to a NetCDF file
    create_nc!(ncresult, ["lon", "lat"], [lon_size, lat_size])  
    append_nc!(ncresult, "lat", lat_values[axes(reflectance,2)[px_range_i]], Dict("latitude" => "latitude"), ["lat"])
    append_nc!(ncresult, "lon", lon_values[axes(reflectance,1)[px_range_j]], Dict("longitude" => "longitude"), ["lon"])
    append_nc!(ncresult, "chl", mat_chl, Dict("chl" => "chl"), ["lon", "lat"])
    append_nc!(ncresult, "lai", mat_lai, Dict("lai" => "lai"), ["lon", "lat"])
    append_nc!(ncresult, "lma", mat_cbc .+ mat_pro, Dict("lma" => "lma"), ["lon", "lat"])
    append_nc!(ncresult, "lwc", mat_lwc, Dict("lwc" => "lwc"), ["lon", "lat"])
    append_nc!(ncresult, "cbc", mat_cbc, Dict("cbc" => "cbc"), ["lon", "lat"])
    append_nc!(ncresult, "pro", mat_pro, Dict("pro" => "pro"), ["lon", "lat"])

    # Run garbage collection
    GC.gc()

    return fittings
end

# Run the function
datafile = "merged_output_subset.nc"
outputfile = "test_output_rmse.nc"
fit_shift_traits!(datafile, outputfile)
