using NetcdfIO: read_nc, create_nc!, append_nc!
using Glob: glob

function find_indices(sub_coords, global_coords; atol=1e-6)
    idx = Int[]
    for val in sub_coords
        found = findfirst(x -> isapprox(x, val; atol=atol), global_coords)
        if isnothing(found)
            error("Coordinate value $val not found in global grid!")
        end
        push!(idx, found)
    end
    return idx
end

function merge_batches!(batch_dir::String, final_output::String)

    # Get list of batch files
    batch_files = sort(glob("batch_*.nc", batch_dir))

    @info "Merging $(length(batch_files)) batches from $batch_dir"

    # Extract all lat/lon coordinates from batch files
    lat_coords = Float64[]
    lon_coords = Float64[]

    for file in batch_files
        lat = read_nc(Float64, file, "lat")
        lon = read_nc(Float64, file, "lon")
        append!(lat_coords, lat)
        append!(lon_coords, lon)
    end

    # Get sorted, unique lat/lon to build global grid
    global_lat = sort(unique(lat_coords))
    global_lon = sort(unique(lon_coords))

    global_lat_size = length(global_lat)
    global_lon_size = length(global_lon)

    @info "Global grid size: $(global_lat_size)x$(global_lon_size)"

    # Initialize trait matrices
    trait_names = ["chl", "lai", "lma", "lwc", "cbc", "pro"]
    trait_matrices = Dict(trait => fill(NaN, global_lat_size, global_lon_size) for trait in trait_names)

    # Merge data from batches
    for file in batch_files
        @info "Processing $file"

        batch_lat = read_nc(Float64, file, "lat")
        batch_lon = read_nc(Float64, file, "lon")

        # Find indices with tolerance for float precision issues
        lat_idx = find_indices(batch_lat, global_lat)
        lon_idx = find_indices(batch_lon, global_lon)

        batch_lat_size = length(batch_lat)
        batch_lon_size = length(batch_lon)

        @info "Batch lat size: $batch_lat_size  lon size: $batch_lon_size"
        @info "Global lat_idx length: $(length(lat_idx))  lon_idx length: $(length(lon_idx))"

        # Sanity check
        if isempty(lat_idx) || isempty(lon_idx)
            @warn "Empty lat/lon indices! Skipping file $file"
            continue
        end

        for trait in trait_names
            batch_data = read_nc(Float64, file, trait)

            # Transpose batch data if needed
            if size(batch_data) != (batch_lon_size, batch_lat_size)
                @warn "Transposing data from $file for $trait (Expected $(batch_lon_size),$(batch_lat_size), got $(size(batch_data)))"
                batch_data = batch_data'
            end

            # Final sanity check before assignment
            if size(batch_data) != (batch_lat_size, batch_lon_size)
                error("Cannot assign data from $file for trait $trait. Shape $(size(batch_data)) does not match lat/lon indices $(batch_lat_size)x$(batch_lon_size)")
            end

            # Assign batch data into global matrix
            trait_matrices[trait][lat_idx, lon_idx] = batch_data
        end
    end

    # Write merged output NetCDF
    @info "Writing merged file to $final_output"

    create_nc!(final_output, ["lon", "lat"], [global_lon_size, global_lat_size])
    append_nc!(final_output, "lat", global_lat, Dict("units" => "degrees_north"), ["lat"])
    append_nc!(final_output, "lon", global_lon, Dict("units" => "degrees_east"), ["lon"])

    for trait in trait_names
        append_nc!(final_output,
                   trait,
                   trait_matrices[trait]',  # Transpose back to lon/lat
                   Dict("units" => "unknown"),
                   ["lon", "lat"])
    end

    @info "✅ Merging complete! Output saved to $final_output"
end

# Example usage
batch_dir = "data/test_batches_scaled"
final_output = "data/test_batches_scaled/merged_output.nc"

merge_batches!(batch_dir, final_output)
