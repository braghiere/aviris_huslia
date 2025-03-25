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

num_workers = min(48, Sys.CPU_THREADS ÷ 2)
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

# This function stays the same as in the previous implementation
function process_batch!(datafile::String, batch_i::Int, batch_j::Int, batch_size_lat::Int, batch_size_lon::Int, batch_dir::String)
    wavelengths = read_nc(Float64, datafile, "wavelength")
    reflectance = read_nc(Float64, datafile, "Reflectance")
    lat_values  = read_nc(Float64, datafile, "lat")  
    lon_values  = read_nc(Float64, datafile, "lon")  

    reflectance = clamp.(reflectance, 0.0, 1.0)

    lat_size, lon_size = size(reflectance)[1:2]
    sub_i = batch_i:min(batch_i + batch_size_lat - 1, lat_size)
    sub_j = batch_j:min(batch_j + batch_size_lon - 1, lon_size)

    row = ceil(Int, batch_i / batch_size_lat)
    col = ceil(Int, batch_j / batch_size_lon)

    batch_file = @sprintf("%s/batch_%02d_%02d.nc", batch_dir, row, col)

    if isfile(batch_file)
        @info "Skipping completed batch: $batch_file"
        return
    end

    params = []
    valid_indices = []

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

    fittings = @showprogress pmap(fit_shift_traits, params; batch_size=10)

    trait_names = ["chl", "lai", "lma", "lwc", "cbc", "pro"]
    batch_lat_size = length(sub_i)
    batch_lon_size = length(sub_j)

    trait_matrices = Dict(trait => fill(NaN, batch_lat_size, batch_lon_size) for trait in trait_names)

    for (idx, (i, j)) in enumerate(valid_indices)
        i_local = i - batch_i + 1
        j_local = j - batch_j + 1
        for (trait_idx, trait) in enumerate(trait_names)
            trait_matrices[trait][i_local, j_local] = fittings[idx][trait_idx]
        end
        trait_matrices["lma"][i_local, j_local] = fittings[idx][5] + fittings[idx][6]
    end

    create_nc!(batch_file, ["lon", "lat"], [batch_lon_size, batch_lat_size])  
    append_nc!(batch_file, "lat", lat_values[sub_i], Dict("latitude" => "latitude"), ["lat"])
    append_nc!(batch_file, "lon", lon_values[sub_j], Dict("longitude" => "longitude"), ["lon"])

    for trait in trait_names
        append_nc!(batch_file, trait, trait_matrices[trait], Dict(trait => trait), ["lon", "lat"])
    end

    GC.gc()

    @info "✅ Batch saved: $batch_file"
end

# ----- Main Execution -----
batch_dir = "data/test_batches_4"
datafile = "data/merged_output.nc"

# Batch sizes for 10x10 grid, but we'll only process the first 2x2 batches
const BATCH_SIZE_LAT = 589  # 5890 / 10
const BATCH_SIZE_LON = 66   # 660 / 10

mkpath(batch_dir)

# Process first 4 batches (01_01, 01_02, 02_01, 02_02)
@info "Running test on first 4 batches only..."

i_batches = [1, BATCH_SIZE_LAT + 1]  # Starting indices for lat: 1, 590
j_batches = [1, BATCH_SIZE_LON + 1]  # Starting indices for lon: 1, 67

for i_batch in i_batches
    for j_batch in j_batches
        process_batch!(datafile, i_batch, j_batch, BATCH_SIZE_LAT, BATCH_SIZE_LON, batch_dir)
    end
end

@info "✅ Test for 4 batches complete!"
