#!/bin/bash
# Run script for EKI v4 with LiDAR constraints
# Full domain: Reflectance → Traits → GPP
# NOW WITH LAI AND CI FROM LIDAR!

echo "=========================================="
echo "EKI v4: Full Domain with LiDAR Constraints"
echo "=========================================="
echo "✨ LAI from LiDAR map"
echo "✨ CI from LiDAR map"
echo "✨ Only 4 free params: cab, lwc, cbc, pro"
echo "✨ Soil moisture = 0.70 (near saturation)"
echo "=========================================="
echo ""

# Check if LiDAR files exist
LAI_FILE="data/lai_output/netcdf/lai_aviris_grid.nc"
CI_FILE="data/lai_output/netcdf/ci_aviris_grid.nc"

if [ ! -f "$LAI_FILE" ]; then
    echo "❌ ERROR: LAI file not found: $LAI_FILE"
    echo "   Please run the LiDAR processing first!"
    exit 1
fi

if [ ! -f "$CI_FILE" ]; then
    echo "❌ ERROR: CI file not found: $CI_FILE"
    echo "   Please run the LiDAR processing first!"
    exit 1
fi

echo "✓ LAI NetCDF found: $LAI_FILE"
echo "✓ CI NetCDF found: $CI_FILE"
echo ""

# Set number of threads
export JULIA_NUM_THREADS=80
echo "Julia threads: $JULIA_NUM_THREADS"
echo ""

# Log file
LOGFILE="logs/eki_v4_full_domain_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "Starting processing..."
echo "Log file: $LOGFILE"
echo ""

# Run with nohup so it continues even if SSH disconnects
nohup /home/renatob/julia-1.10.0/bin/julia --project full_domain_refl_to_gpp_eki_v4.jl > "$LOGFILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo ""
echo "📋 To monitor progress:"
echo "   tail -f $LOGFILE"
echo ""
echo "📋 To check if running:"
echo "   ps -p $PID"
echo ""
echo "📋 To stop:"
echo "   kill $PID"
echo ""
echo "📋 Output will be in:"
echo "   data/output_full_domain_traits_gpp_eki_v4.nc"
echo ""
echo "📋 Checkpoint file:"
echo "   data/processing_checkpoint_eki_v4.json"
echo ""
echo "=========================================="
echo "🚀 EKI v4 processing launched!"
echo "=========================================="
