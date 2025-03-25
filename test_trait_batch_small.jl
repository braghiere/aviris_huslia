#########################################################################################
# 🌎 Hyperspectral Trait Extraction & NetCDF Export (CF-1.6 compliant)
#########################################################################################

# --------------------------------------------------
# 📦 Imports
# --------------------------------------------------
using NetcdfIO: append_nc!, create_nc!, read_nc
using Distributed
using ProgressMeter: @showprogress
using Base.GC
using Printf
using Statistics
using LinearAlgebra: transpose
using Dates
using NetCDF

#########################################################################################
# ⚙️ Parallel Worker Management
#########################################################################################

# Clean existing workers (optional)
if length(workers()) > 1
    rmprocs(workers())
end

# Add workers (change as needed)
num_workers = min(40, Sys.CPU_THREADS ÷ 2)
addprocs(num_workers; exeflags="--project")

@info "✅ Using $num_workers parallel workers."

# -------------------------------------------------------------------
# ✅ Sync environment and load dependencies on workers
# -------------------------------------------------------------------

# Set up the project environment and instantiate packages
#@everywhere using Pkg
#@everywhere Pkg.activate(".")
#@everywhere Pkg.instantiate()

# Load required packages on all workers (one-liners, no `begin` blocks!)
@everywhere using NetCDF
@everywhere include("target_function_v3.jl")

#########################################################################################
# 🌱 Helper Functions
#########################################################################################

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
    transpose_for_nc(mat)

Transpose matrix for NetCDF (lon, lat).
"""
transpose_for_nc(mat) = permutedims(mat, (2, 1))

#########################################################################################
# 🔨 Batch Processing Function
#########################################################################################

"""
    process_batch!(datafile, batch_i, batch_j, batch_size_lat, batch_size_lon, batch_dir)

Processes a batch and saves the result as a CF-compliant NetCDF file.
"""
function process_batch!(datafile::String, batch_i::Int, batch_j::Int,
                        batch_size_lat::Int, batch_size_lon::Int, batch_dir::String)

    # --- Load data ---
    wavelengths = read_nc(Float64, datafile, "wavelength")
    reflectance = read_nc(Float64, datafile, "Reflectance")
    lat_values  = read_nc(Float64, datafile, "lat")  
    lon_values  = read_nc(Float64, datafile, "lon")  

    reflectance = clamp.(reflectance, 0.0, 1.0)

    # --- Define subgrid ---
    lat_size, lon_size = size(reflectance)[1:2]
    sub_i = batch_i:min(batch_i + batch_size_lat - 1, lat_size)
    sub_j = batch_j:min(batch_j + batch_size_lon - 1, lon_size)

    row = ceil(Int, batch_i / batch_size_lat)
    col = ceil(Int, batch_j / batch_size_lon)

    batch_file = @sprintf("%s/batch_%02d_%02d.nc", batch_dir, row, col)

    if isfile(batch_file)
        @info "⏭️  Skipping completed batch: $batch_file"
        return
    end

    # --- Collect valid pixels ---
    params = []
    valid_indices = []

    for i in sub_i, j in sub_j
        ndvi = compute_ndvi(reflectance[i, j, :], wavelengths)
        if ndvi > 0.2
            push!(params, (deepcopy(wavelengths), reflectance[i, j, :], i, j))
            push!(valid_indices, (i, j))
        end
    end

    # --- Initialize matrices ---
    trait_names = ["chl", "lai", "lma", "lwc", "cbc", "pro"]
    batch_lat_size = length(sub_i)
    batch_lon_size = length(sub_j)

    trait_matrices = Dict(trait => fill(NaN, batch_lat_size, batch_lon_size) for trait in trait_names)

    if isempty(params)
        @info "⚠️ No valid pixels in batch row $(row) col $(col). Writing NaN-filled batch..."
    else
        @info "🚀 Processing batch row $(row) col $(col) with $(length(params)) valid pixels."

        # --- Run trait fitting in parallel ---
        fittings = @showprogress pmap(fit_shift_traits, params; batch_size=5)

        # --- Fill matrices ---
        for (idx, (i, j)) in enumerate(valid_indices)
            i_local = i - batch_i + 1
            j_local = j - batch_j + 1

            for (trait_idx, trait) in enumerate(trait_names)
                trait_matrices[trait][i_local, j_local] = fittings[idx][trait_idx]
            end

            # Combine cbc + pro into lma if needed
            trait_matrices["lma"][i_local, j_local] = fittings[idx][5] + fittings[idx][6]
        end
    end

    # --- Write CF-1.6 compliant NetCDF ---
    cf_global_atts = Dict(
        "Conventions" => "CF-1.6",
        "title" => "AVIRIS Hyperspectral Trait Product",
        "source" => "Hyperspectral reflectance data processed with Julia",
        "history" => "Created on $(Dates.now())"
    )

    # 1. Create NetCDF structure
    create_nc!(batch_file, ["lon", "lat"], [batch_lon_size, batch_lat_size])

    # 2. Append lat & lon coordinates
    append_nc!(batch_file, "lat", lat_values[sub_i], Dict(
        "units" => "degrees_north",
        "standard_name" => "latitude",
        "long_name" => "Latitude coordinate"
    ), ["lat"])

    append_nc!(batch_file, "lon", lon_values[sub_j], Dict(
        "units" => "degrees_east",
        "standard_name" => "longitude",
        "long_name" => "Longitude coordinate"
    ), ["lon"])

    # 3. Append trait data
    for trait in trait_names
        append_nc!(batch_file,
            trait,
            transpose_for_nc(trait_matrices[trait]),
            Dict(
                "units" => "unknown", # update with actual units
                "standard_name" => trait,
                "long_name" => "Trait: $trait",
                "coordinates" => "lon lat"
            ),
            ["lon", "lat"])
    end

    # 4. Add global attributes and _FillValues using NetCDF.jl
    ds = NCDataset(batch_file, "a")
    for trait in trait_names
        ds[trait].attrib["_FillValue"] = NaN
        ds[trait].attrib["missing_value"] = NaN
    end

    for (att_name, att_value) in cf_global_atts
        ds.attrib[att_name] = att_value
    end
    close(ds)

    GC.gc() # Clean memory
    @info "✅ Batch saved: $batch_file"
end

#########################################################################################
# 🏁 Main Execution
#########################################################################################

batch_dir = "data/test_batches_scaled"
datafile = "data/merged_output_small.nc"

# Dataset dimensions (scaled-down example)
const LAT_SIZE = 589
const LON_SIZE = 66

# Number of batches
const NUM_BATCHES_LAT = 2
const NUM_BATCHES_LON = 2

const BATCH_SIZE_LAT = ceil(Int, LAT_SIZE / NUM_BATCHES_LAT)
const BATCH_SIZE_LON = ceil(Int, LON_SIZE / NUM_BATCHES_LON)

mkpath(batch_dir)

@info "🔬 Running test on first 4 scaled-down batches..."
i_batches = [1, BATCH_SIZE_LAT + 1]
j_batches = [1, BATCH_SIZE_LON + 1]

for i_batch in i_batches
    for j_batch in j_batches
        process_batch!(datafile, i_batch, j_batch, BATCH_SIZE_LAT, BATCH_SIZE_LON, batch_dir)
    end
end

@info "🎉 Test for 4 batches on scaled data complete!"
