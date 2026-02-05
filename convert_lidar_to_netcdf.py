#!/usr/bin/env python3
"""
Convert reprojected LiDAR GeoTIFF files to NetCDF format.
Uses the same lat/lon coordinates as the AVIRIS reflectance data.
"""

import rasterio
import numpy as np
from netCDF4 import Dataset
from datetime import datetime

# Input GeoTIFF files (reprojected to AVIRIS grid)
input_dir = "data/lai_output/reprojected_to_aviris"
output_dir = "data/lai_output/netcdf"

# Input files
lidar_files = {
    'lai': f'{input_dir}/lai_effective_aviris_grid.tif',
    'ci': f'{input_dir}/clumping_index_aviris_grid.tif',
    'true_lai': f'{input_dir}/lai_true_aviris_grid.tif',
    'canopy_cover': f'{input_dir}/canopy_cover_aviris_grid.tif',
    'fhd': f'{input_dir}/fhd_aviris_grid.tif'
}

# Get coordinates from AVIRIS reflectance file
aviris_file = "data/merged_output_cf.nc"
print(f"Reading coordinates from {aviris_file}...")
with Dataset(aviris_file, 'r') as ds:
    lat = ds['lat'][:]
    lon = ds['lon'][:]
    print(f"  Lat: {len(lat)} points ({lat.min():.4f} to {lat.max():.4f})")
    print(f"  Lon: {len(lon)} points ({lon.min():.4f} to {lon.max():.4f})")

# Variable metadata
metadata = {
    'lai': {
        'long_name': 'Effective Leaf Area Index',
        'units': 'm2/m2',
        'description': 'Effective leaf area index from LiDAR canopy height model'
    },
    'ci': {
        'long_name': 'Clumping Index',
        'units': '1',
        'description': 'Canopy clumping index from LiDAR point cloud analysis'
    },
    'true_lai': {
        'long_name': 'True Leaf Area Index',
        'units': 'm2/m2',
        'description': 'True leaf area index (LAI / CI) from LiDAR'
    },
    'canopy_cover': {
        'long_name': 'Canopy Cover',
        'units': '1',
        'description': 'Fractional canopy cover from LiDAR'
    },
    'fhd': {
        'long_name': 'Foliage Height Diversity',
        'units': '1',
        'description': 'Vertical distribution of foliage from LiDAR'
    }
}

# Convert each file
import os
os.makedirs(output_dir, exist_ok=True)

for var_name, input_file in lidar_files.items():
    output_file = f"{output_dir}/{var_name}_aviris_grid.nc"
    
    print(f"\nConverting {var_name}...")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")
    
    # Read GeoTIFF
    with rasterio.open(input_file) as src:
        data = src.read(1)  # Read first band
        profile = src.profile
        
        # GeoTIFF has row 0 at bottom (South), but NetCDF lat array goes North to South
        # So we need to flip the data vertically
        data = np.flipud(data)
        
        print(f"  Data shape: {data.shape} (flipped to match NetCDF lat direction)")
        
        # Count valid pixels
        valid_mask = ~np.isnan(data)
        n_valid = np.sum(valid_mask)
        print(f"  Valid pixels: {n_valid:,} ({100*n_valid/data.size:.1f}%)")
        
        if n_valid > 0:
            print(f"  Value range: {np.nanmin(data):.3f} to {np.nanmax(data):.3f}")
            print(f"  Mean: {np.nanmean(data):.3f}")
    
    # Create NetCDF file
    with Dataset(output_file, 'w', format='NETCDF4') as nc:
        # Create dimensions
        nc.createDimension('lat', len(lat))
        nc.createDimension('lon', len(lon))
        
        # Create coordinate variables
        lat_var = nc.createVariable('lat', 'f8', ('lat',))
        lat_var[:] = lat
        lat_var.long_name = 'latitude'
        lat_var.units = 'degrees_north'
        lat_var.standard_name = 'latitude'
        
        lon_var = nc.createVariable('lon', 'f8', ('lon',))
        lon_var[:] = lon
        lon_var.long_name = 'longitude'
        lon_var.units = 'degrees_east'
        lon_var.standard_name = 'longitude'
        
        # Create data variable
        var = nc.createVariable(var_name, 'f4', ('lat', 'lon'), 
                               fill_value=np.nan, zlib=True, complevel=4)
        var[:] = data
        var.long_name = metadata[var_name]['long_name']
        var.units = metadata[var_name]['units']
        var.description = metadata[var_name]['description']
        var.coordinates = 'lat lon'
        var.grid_mapping = 'crs'
        
        # Add CRS information
        crs = nc.createVariable('crs', 'i4')
        crs.grid_mapping_name = 'latitude_longitude'
        crs.longitude_of_prime_meridian = 0.0
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        crs.spatial_ref = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        crs.crs_wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        
        # Global attributes
        nc.Conventions = 'CF-1.8'
        nc.title = f'LiDAR-derived {metadata[var_name]["long_name"]}'
        nc.institution = 'California Institute of Technology'
        nc.source = 'Processed from airborne LiDAR point cloud data'
        nc.history = f'{datetime.now().isoformat()}: Converted from GeoTIFF to NetCDF, reprojected to AVIRIS grid'
        nc.references = 'Huslia, Alaska LiDAR campaign'
        nc.comment = f'Reprojected from UTM Zone 4N (EPSG:32604) to WGS84 (EPSG:4326) to match AVIRIS reflectance grid'
    
    print(f"  ✓ Saved to {output_file}")

print("\n" + "="*70)
print("✓ All LiDAR files converted to NetCDF format")
print(f"  Output directory: {output_dir}")
print("="*70)
