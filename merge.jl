using NetCDF
using Glob
using Printf

# === CONFIGURATION ===
batch_dir = "/net/fluo/data3/data/FluoData1/students/renato/aviris_huslia/data/batches"
file_pattern = joinpath(batch_dir, "batch_*.nc")

# Global grid shape
nlat = 5890
nlon = 660

# Define your batch chunk sizes (assume 50x50 unless you specify differently)
lat_chunk = 50
lon_chunk = 50

# Compute number of rows/columns
num_rows = ceil(Int, nlat / lat_chunk)
num_cols = ceil(Int, nlon / lon_chunk)

println("Global grid: $(nlat)x$(nlon)")
println("Chunk size: $(lat_chunk)x$(lon_chunk)")
println("Expecting: $(num_rows) rows x $(num_cols) columns")

# === FUNCTION TO MAP FILE BY GRID POSITION ===
function find_grid_position(lat_val, lon_val, merged_lat, merged_lon)
    # Find the closest indices in the global merged lat/lon grids
    lat_idx = findmin(abs.(merged_lat .- lat_val))[2]
    lon_idx = findmin(abs.(merged_lon .- lon_val))[2]

    # Map to row/column
    row = ceil(Int, lat_idx / lat_chunk)
    col = ceil(Int, lon_idx / lon_chunk)

    return row, col
end

# === LOAD MERGED GRID LAT/LON ===
merged_file = "/net/fluo/data3/data/FluoData1/students/renato/aviris_huslia/data/merged_output.nc"

println("🔎 Loading merged lat/lon from merged_output.nc...")

ds_merged = NetCDF.open(merged_file)
merged_lat = ds_merged["lat"][:, 1]   # Assuming lat varies along first dimension
merged_lon = ds_merged["lon"][1, :]   # Assuming lon varies along second dimension
NetCDF.close(ds_merged)

println("  ➡️ Loaded merged lat size: ", length(merged_lat))
println("  ➡️ Loaded merged lon size: ", length(merged_lon))

# === PROCESS EACH BATCH ===
batch_files = glob("batch_*.nc", batch_dir)

println("🔎 Found $(length(batch_files)) batch files in $(batch_dir)")

for file in batch_files
    println("\n🔹 Processing: $(basename(file))")

    ds = NetCDF.open(file)

    # 1D lat/lon variables
    lats = ds["lat"][:]
    lons = ds["lon"][:]

    # Get top-left corner lat/lon (or any specific point)
    min_lat = lats[1]
    min_lon = lons[1]

    println("  ➡️ Top-left corner lat/lon: ($min_lat, $min_lon)")

    # Map lat/lon to row/col in the global grid
    row, col = find_grid_position(min_lat, min_lon, merged_lat, merged_lon)

    println("  ➡️ Mapped to row: $row, col: $col")

    # Create new filename
    new_filename = @sprintf("batch_%02d_%02d.nc", row, col)
    new_filepath = joinpath(batch_dir, new_filename)

    println("  ➡️ Renaming to: $new_filename")

    # Rename file
    mv(file, new_filepath; force=true)

    NetCDF.close(ds)
end

println("\n✅ All files renamed by row/col grid position.")
