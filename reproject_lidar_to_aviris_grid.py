#!/usr/bin/env python3
"""
Reproject LiDAR LAI and CI maps to match AVIRIS reflectance grid
This creates versions of the maps that can be directly indexed with the same i,j as reflectance
"""

import rasterio
from rasterio.warp import reproject, Resampling
import netCDF4 as nc
import numpy as np
from pathlib import Path

print("="*80)
print("Reprojecting LiDAR maps to AVIRIS grid")
print("="*80)

# Input files - use ORIGINAL mosaics with correct CRS (EPSG:32604)
lidar_dir = Path("data/lai_output")
lai_file = lidar_dir / "Huslia_mosaic_lai_effective.tif"
ci_file = lidar_dir / "Huslia_mosaic_omega.tif"
fhd_file = lidar_dir / "Huslia_mosaic_fhd.tif"
cover_file = lidar_dir / "Huslia_mosaic_canopy_cover.tif"
lai_true_file = lidar_dir / "Huslia_mosaic_lai_true.tif"

# Reflectance file to get target grid
refl_file = "data/merged_output_cf.nc"

# Output directory
output_dir = Path("data/lai_output/reprojected_to_aviris")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📂 Reading AVIRIS grid from: {refl_file}")
with nc.Dataset(refl_file) as ncf:
    lats = ncf.variables['lat'][:]
    lons = ncf.variables['lon'][:]
    
    lat_size, lon_size = len(lats), len(lons)
    print(f"   AVIRIS grid size: {lon_size} x {lat_size} (lon x lat)")
    print(f"   Lat range: {lats.min():.4f} to {lats.max():.4f}")
    print(f"   Lon range: {lons.min():.4f} to {lons.max():.4f}")

# Create target geotransform for AVIRIS grid
# Note: AVIRIS uses center coordinates, we need corner coordinates
dlat = lats[1] - lats[0] if len(lats) > 1 else 0.0001
dlon = lons[1] - lons[0] if len(lons) > 1 else 0.0001

# Corner coordinates (top-left)
lon_min = lons[0] - dlon/2
lat_max = lats[-1] + dlat/2  # lats are ascending in AVIRIS

from rasterio.transform import from_bounds
target_transform = from_bounds(
    lon_min, 
    lats[0] - dlat/2,  # bottom
    lons[-1] + dlon/2,  # right
    lat_max,  # top
    lon_size, 
    lat_size
)

target_crs = 'EPSG:4326'  # WGS84

print(f"\n📊 Target transform:")
print(f"   Pixel size: {dlon:.6f} x {dlat:.6f} degrees")
print(f"   CRS: {target_crs}")

def reproject_lidar_to_aviris(input_file, output_file, description):
    """Reproject a LiDAR GeoTIFF to AVIRIS grid"""
    print(f"\n🔄 Processing {description}...")
    print(f"   Input:  {input_file.name}")
    print(f"   Output: {output_file.name}")
    
    with rasterio.open(input_file) as src:
        print(f"   Source: {src.width} x {src.height}, CRS: {src.crs}")
        
        # Create output array
        output_data = np.full((lat_size, lon_size), np.nan, dtype=np.float32)
        
        # Reproject
        reproject(
            source=rasterio.band(src, 1),
            destination=output_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.average,
            src_nodata=src.nodata,
            dst_nodata=np.nan
        )
        
        # Write output
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=lat_size,
            width=lon_size,
            count=1,
            dtype=np.float32,
            crs=target_crs,
            transform=target_transform,
            nodata=np.nan,
            compress='lzw'
        ) as dst:
            dst.write(output_data, 1)
            dst.set_band_description(1, description)
        
        valid_pixels = np.sum(~np.isnan(output_data))
        mean_val = np.nanmean(output_data)
        print(f"   ✓ Valid pixels: {valid_pixels:,} / {output_data.size:,}")
        print(f"   ✓ Mean value: {mean_val:.3f}")
        
        return output_data

# Reproject all LiDAR products
print("\n" + "="*80)
print("Reprojecting LiDAR products...")
print("="*80)

lai_eff = reproject_lidar_to_aviris(
    lai_file, 
    output_dir / "lai_effective_aviris_grid.tif",
    "Effective LAI (reprojected to AVIRIS grid)"
)

ci = reproject_lidar_to_aviris(
    ci_file,
    output_dir / "clumping_index_aviris_grid.tif", 
    "Clumping Index (reprojected to AVIRIS grid)"
)

fhd = reproject_lidar_to_aviris(
    fhd_file,
    output_dir / "fhd_aviris_grid.tif",
    "Foliage Height Diversity (reprojected to AVIRIS grid)"
)

cover = reproject_lidar_to_aviris(
    cover_file,
    output_dir / "canopy_cover_aviris_grid.tif",
    "Canopy Cover (reprojected to AVIRIS grid)"
)

lai_true = reproject_lidar_to_aviris(
    lai_true_file,
    output_dir / "lai_true_aviris_grid.tif",
    "True LAI (reprojected to AVIRIS grid)"
)

print("\n" + "="*80)
print("✅ All LiDAR products reprojected successfully!")
print("="*80)
print(f"\nOutput directory: {output_dir}")
print("\nFiles created:")
print(f"  - lai_effective_aviris_grid.tif")
print(f"  - clumping_index_aviris_grid.tif")
print(f"  - fhd_aviris_grid.tif")
print(f"  - canopy_cover_aviris_grid.tif")
print(f"  - lai_true_aviris_grid.tif")
print("\n✨ These files can now be directly indexed with reflectance i,j indices!")
