using NetcdfIO: append_nc!, create_nc!, read_nc
# using Emerald.EmeraldVisualization: canvas, save_canvas!

using Distributed: @everywhere, addprocs, pmap, workers
using ProgressMeter: @showprogress

using Base.GC

using Printf

# you can change the number of threads here
# Fluo: within 48
# Tofu: within 48
# Curry: within 256
if length(workers()) == 1
    #addprocs(1; exeflags = "--project");
    #addprocs(72; exeflags = "--project");
    addprocs(148; exeflags = "--project");
end;

@everywhere include("target_function.jl");

function fit_shift_traits!(datafile::String, datafile_lai::String, datafile_ci::String, ncresult::String)
    _wlen = read_nc(FT, datafile, "wavelength")
    _refl = read_nc(FT, datafile, "reflectance")
    _lat_values = read_nc(FT, datafile, "latitude")  # Assuming 'latitude' is the variable name
    _lon_values = read_nc(FT, datafile, "longitude")  # Assuming 'longitude' is the variable name
    
    #_lai = read_nc(FT, datafile_lai, "lai")
    #_ci = read_nc(FT, datafile_ci, "ci")

    # Clip reflectance data to be within the range [0, 1]
    _refl = clamp.(_refl, 0.0, 1.0)

    # create a vector of parameters
    _params = []
    #for _i in axes(_refl,1)[1600:1610], _j in axes(_refl,2)[1600:1610] # change the index number of the lat and lon
    for _i in axes(_refl, 1), _j in axes(_refl, 2) # change the index number of the lat and lon
        #push!(_params, (deepcopy(_wlen), _refl[_i, _j, :, 1], _i, _j, _lai[_i, _j, 1], _ci[_i, _j]))
        push!(_params, (deepcopy(_wlen), _refl[_i, _j, :, 1], _i, _j,))
    end

    _fittings = (@showprogress pmap(fit_shift_traits, _params))
    _chls = [_f[1] for _f in _fittings]
    _lais = [_f[2] for _f in _fittings]
    _lmas = [_f[3] for _f in _fittings]
    _lwcs = [_f[4] for _f in _fittings]
    _cbcs = [_f[5] for _f in _fittings]
    _pros = [_f[6] for _f in _fittings]

    _len_lat = size(_refl, 2)
    _len_lon = size(_refl, 1)
    #_len_lat = length(axes(_refl,2)[1600:1610]);
    #_len_lon = length(axes(_refl,1)[1600:1610]);
    _mat_chls = reshape(_chls, _len_lat, _len_lon)
    _mat_lais = reshape(_lais, _len_lat, _len_lon)
    _mat_lmas = reshape(_lmas, _len_lat, _len_lon)
    _mat_lwcs = reshape(_lwcs, _len_lat, _len_lon)
    _mat_cbcs = reshape(_cbcs, _len_lat, _len_lon)
    _mat_pros = reshape(_pros, _len_lat, _len_lon)

    # save the matrices to a file
    create_nc!(ncresult, String["lat", "lon"], [_len_lat, _len_lon])  
    append_nc!(ncresult, "lat", _lat_values, Dict("latitude" => "latitude"), ["lat"])
    append_nc!(ncresult, "lon", _lon_values, Dict("longitude" => "longitude"), ["lon"])
    #append_nc!(ncresult, "lat", _lat_values[axes(_refl,2)[1600:1610]], Dict("latitude" => "latitude"), ["lat"]);
    #append_nc!(ncresult, "lon", _lon_values[axes(_refl,1)[1600:1610]], Dict("longitude" => "longitude"), ["lon"]);
    append_nc!(ncresult, "chl", _mat_chls, Dict("chl" => "chl"), ["lat", "lon"])
    append_nc!(ncresult, "lai", _mat_lais, Dict("lai" => "lai"), ["lat", "lon"])
    append_nc!(ncresult, "lma", _mat_cbcs + _mat_pros, Dict("lma" => "lma"), ["lat", "lon"])
    append_nc!(ncresult, "lwc", _mat_lwcs, Dict("lwc" => "lwc"), ["lat", "lon"])
    append_nc!(ncresult, "cbc", _mat_cbcs, Dict("cbc" => "cbc"), ["lat", "lon"])
    append_nc!(ncresult, "pro", _mat_pros, Dict("pro" => "pro"), ["lat", "lon"])
    #append_nc!(ncresult, "ci", _mat_cis, Dict("ci" => "ci"), ["lon", "lat"])
    
    GC.gc()

    return _fittings
end

# Loop over files from time_00 to time_12
function process_all_times()
    base_datafile = "/net/fluo/data3/data/FluoData1/students/renato/aviris_huslia/merged_output.nc"
    base_ncresult = "huslia_fittings.nc"
    
    #for time in 1:6
    for time in 0:1
        #datafile = @sprintf(base_datafile, time)
        datafile = @sprintf("/net/fluo/data3/data/FluoData1/students/renato/aviris_huslia/merged_output.nc", time)
        #datafile_lai = @sprintf("/net/squid/data3/data/FluoData1/students/renato/aviris_dangermond/traits/datasets/index/lai_aviris_dangermond_time_%02d.nc", time)
        #datafile_ci = @sprintf("/net/fluo/data3/data/FluoData1/students/renato/aviris_dangermond/clumping_index/GEDI_Dangermonfiles20240610043741/GEDI_CI_map_ci_nan.nc")
        #ncresult = @sprintf(base_ncresult, time)
        ncresult = @sprintf("huslia_fittings.nc", time)
        fit_shift_traits!(datafile, datafile_lai, datafile_ci, ncresult)
        
        GC.gc()
    end
end

# Run the function to process all times
process_all_times()
