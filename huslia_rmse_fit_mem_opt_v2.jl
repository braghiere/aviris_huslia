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

num_workers = min(96, Sys.CPU_THREADS ÷ 2)  # Use half of CPU threads (max 24)
if length(workers()) == 1
    addprocs(num_workers; exeflags = "--project")
end

@info "Using $num_workers parallel workers."

# Load the target function for spectral fitting
@everywhere include("target_function_v3.jl")

"""
    compute_ndvi(reflectance, wavelengths)

Computes NDVI using Red (660-680 nm) and NIR (840-860 nm) bands.
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
    process_batch!(datafile, batch_i, batch_j, batch_size, batch_dir)

Processes a single batch and saves results as a NetCDF file.
"""
function process_batch!(datafile::String, batch_i::Int, batch_j::Int, batch_size::Int, batch_dir::String)
    # Read input data
    wavelengths = read_nc(Float64, datafile, "wavelength")
    reflectance = read_nc(Float64, datafile, "Reflectance")
    lat_values  = read_nc(Float64, datafile, "lat")  
    lon_values  = read_nc(Float64, datafile, "lon")  

    reflectance = clamp.(reflectance, 0.0, 1.0)

    lat_size, lon_size = size(reflectance)[1:2]
    sub_i = batch_i:min(batch_i + batch_size - 1, lat_size)
    sub_j = batch_j:min(batch_j + batch_size - 1, lon_size)

    # Calculate row and col tile numbers for naming
    row = ceil(Int, batch_i / batch_size)
    col = ceil(Int, batch_j / batch_size)

    batch_file = @sprintf("%s/batch_%02d_%02d.nc", batch_dir, row, col)

    # Skip if batch file already exists
    if isfile(batch_file)
        @info "Skipping completed batch: $batch_file"
        return
    end

    params = []
    valid_indices = []

    # Loop over sub-batch and collect valid NDVI pixels
    for i in sub_i, j in sub_j
        ndvi = compute_ndvi(reflectance[i, j, :], wavelengths)
        if ndvi > 0.2
            push!(params, (deepcopy(wavelengths), reflectance[i, j, :], i, j))
            push!(valid_indices, (i, j))
        end
    end

    if isempty(params)
        @info "No valid pixels in batch row $(row) col $(col), skipping..."
        return
    end

    @info "Processing batch row $(row) col $(col) with $(length(params)) valid pixels."

    # Fit traits in parallel
    fittings = @showprogress pmap(fit_shift_traits, params; batch_size=10)

    # Prepare matrices for traits
    trait_names = ["chl", "lai", "lma", "lwc", "cbc", "pro"]
    batch_lat_size = length(sub_i)
    batch_lon_size = length(sub_j)

    trait_matrices = Dict(trait => fill(NaN, batch_lat_size, batch_lon_size) for trait in trait_names)

    # Populate matrices with fitted values
    for (idx, (i, j)) in enumerate(valid_indices)
        i_local = i - batch_i + 1
        j_local = j - batch_j + 1
        for (trait_idx, trait) in enumerate(trait_names)
            trait_matrices[trait][i_local, j_local] = fittings[idx][trait_idx]
        end
        # Adding cbc + pro for lma specifically
        trait_matrices["lma"][i_local, j_local] = fittings[idx][5] + fittings[idx][6]
    end

    # Save to NetCDF
    create_nc!(batch_file, ["lon", "lat"], [batch_lon_size, batch_lat_size])  
    append_nc!(batch_file, "lat", lat_values[sub_i], Dict("latitude" => "latitude"), ["lat"])
    append_nc!(batch_file, "lon", lon_values[sub_j], Dict("longitude" => "longitude"), ["lon"])

    for trait in trait_names
        append_nc!(batch_file, trait, trait_matrices[trait], Dict(trait => trait), ["lon", "lat"])
    end

    GC.gc()

    @info "✅ Batch saved: $batch_file"
end

"""
    process_all_batches!(datafile, batch_size, batch_dir)

Splits data into smaller batches and processes them individually.
"""
function process_all_batches!(datafile::String, batch_size::Int, batch_dir::String)
    lat_size = length(read_nc(Float64, datafile, "lat"))
    lon_size = length(read_nc(Float64, datafile, "lon"))

    mkpath(batch_dir)  # Ensure batch directory exists

    @info "Starting batch processing with batch size $batch_size on a grid of $(lat_size)x$(lon_size)..."

    for i_batch in 1:batch_size:lat_size
        for j_batch in 1:batch_size:lon_size
            process_batch!(datafile, i_batch, j_batch, batch_size, batch_dir)
        end
    end

    @info "✅ All batches processed! Ready to merge."
end

"""
    merge_batches!(batch_dir::String, final_output::String)

Merges all batch NetCDF files into a single final output file.
"""
function merge_batches!(batch_dir::String, final_output::String)
    batch_files = filter(x -> endswith(x, ".nc"), readdir(batch_dir, join=true))
    if isempty(batch_files)
        error("No batch files found in $batch_dir!")
    end

    @info "Merging $(length(batch_files)) batch files into final output..."

    # Read the first batch to get dimensions
    sample_nc = batch_files[1]
    lat_values = read_nc(Float64, sample_nc, "lat")
    lon_values = read_nc(Float64, sample_nc, "lon")

    lat_size, lon_size = length(lat_values), length(lon_values)

    trait_names = ["chl", "lai", "lma", "lwc", "cbc", "pro"]
    trait_matrices = Dict(trait => fill(NaN, lat_size, lon_size) for trait in trait_names)

    for batch_file in batch_files
        @info "Merging: $batch_file"

        batch_lat = read_nc(Float64, batch_file, "lat")
        batch_lon = read_nc(Float64, batch_file, "lon")

        batch_i = findfirst(==(batch_lat[1]), lat_values)
        batch_j = findfirst(==(batch_lon[1]), lon_values)

        if batch_i === nothing || batch_j === nothing
            @warn "Batch coordinates not aligned: $batch_file"
            continue
        end

        for trait in trait_names
            batch_data = read_nc(Float64, batch_file, trait)
            lat_range = batch_i : batch_i + size(batch_data, 1) - 1
            lon_range = batch_j : batch_j + size(batch_data, 2) - 1

            trait_matrices[trait][lat_range, lon_range] .= batch_data
        end
    end

    # Write the merged NetCDF
    create_nc!(final_output, ["lon", "lat"], [lon_size, lat_size])  
    append_nc!(final_output, "lat", lat_values, Dict("latitude" => "latitude"), ["lat"])
    append_nc!(final_output, "lon", lon_values, Dict("longitude" => "longitude"), ["lon"])

    for trait in trait_names
        append_nc!(final_output, trait, trait_matrices[trait], Dict(trait => trait), ["lon", "lat"])
    end

    @info "✅ Merging complete! Final output saved to $final_output"
end

# ----- Main Execution -----
batch_dir = "data/batches"
datafile = "data/merged_output.nc"
final_output = "data/output_final.nc"

const BATCH_SIZE = 50

process_all_batches!(datafile, BATCH_SIZE, batch_dir)
merge_batches!(batch_dir, final_output)
