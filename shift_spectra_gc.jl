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
    addprocs(1; exeflags = "--project");
end;

@everywhere include("target_function_v3.jl");

# Function to fit the shift traits with multiple threadings
#function fit_shift_traits!(datafile::String = "/net/fluo/data1/students/renato/aviris_dangermond/aviris_dangermond_point_conception_time_00.nc", ncresult::String = "shift_fittings1.nc")
#fit_shift_traits!("/net/fluo/data3/data/FluoData1/students/renato/aviris_dangermond/aviris_dangermond_time_00/output_clipped.nc", "shift_fittings1_test.nc")
#fit_shift_traits!("/net/fluo/data3/data/FluoData1/students/renato/aviris_dangermond/aviris_dangermond_point_conception_time_00.nc", "shift_fittings1_test.nc")
function fit_shift_traits!(datafile::String = "/net/fluo/data3/data/FluoData1/students/renato/aviris_huslia/merged_output.nc", ncresult::String = "test_output.nc")
    _wlen = read_nc(FT, datafile, "wavelength");
    _refl = read_nc(FT, datafile, "Reflectance");
    _lat_values = read_nc(FT, datafile, "lat");  # Assuming 'latitude' is the variable name
    _lon_values = read_nc(FT, datafile, "lon");  # Assuming 'longitude' is the variable name


    # Clip reflectance data to be within the range [0, 1]
    _refl = clamp.(_refl, 0.0, 1.0)


    # create a vector of parameters
    _params = [];
    #for _i in axes(_refl,1)[200:210], _j in axes(_refl,3)[200:210] # change the index number of the lat and lon
    for _i in axes(_refl,1)[330:340], _j in axes(_refl,2)[330:340] # change the index number of the lat and lon
    #for _i in axes(_refl,1), _j in axes(_refl,2) # change the index number of the lat and lon
        #push!(_params, (deepcopy(_wlen), _refl[_i,:,_j,1]));
        #push!(_params, (deepcopy(_wlen), _refl[_i,_j,:,1]));
        push!(_params, (deepcopy(_wlen), _refl[_i,_j,:],_i,_j));
    end;

    _fittings = (@showprogress pmap(fit_shift_traits, _params));
    _chls = [_f[1] for _f in _fittings];
    _lais = [_f[2] for _f in _fittings];
    _lmas = [_f[3] for _f in _fittings];
    _lwcs = [_f[4] for _f in _fittings];
    _cbcs = [_f[5] for _f in _fittings];
    _pros = [_f[6] for _f in _fittings];

    #_len_lat = 11; # size(_refl, 1);
    #_len_lon = 11; # size(_refl, 3);
    #_len_lat = size(_refl, 2); # 
    #_len_lon = size(_refl, 1); # 
    _len_lat = length(axes(_refl,2)[330:340]);
    _len_lon = length(axes(_refl,1)[330:340]);
    _mat_chls = reshape(_chls, _len_lat, _len_lon);
    _mat_lais = reshape(_lais, _len_lat, _len_lon);
    _mat_lmas = reshape(_lmas, _len_lat, _len_lon);
    _mat_lwcs = reshape(_lwcs, _len_lat, _len_lon);
    _mat_cbcs = reshape(_cbcs, _len_lat, _len_lon);
    _mat_pros = reshape(_pros, _len_lat, _len_lon);
    
    
    #println(collect(1:_len_lat))
    #println(axes(_refl,1)[1600:1610])

    #println(_refl[1600:1610,:, :, :])  # Adjust slicing based on the dimensions of _refl
    #println(collect(axes(_refl,1)[200:210]))
    

    # save the matrices to a file
    create_nc!(ncresult, String["lon", "lat"], [_len_lon, _len_lat]);  
    append_nc!(ncresult, "lat", _lat_values[axes(_refl,2)[330:340]], Dict("latitude" => "latitude"), ["lat"]);
    append_nc!(ncresult, "lon", _lon_values[axes(_refl,1)[330:340]], Dict("longitude" => "longitude"), ["lon"]);
    #append_nc!(ncresult, "lat", _lat_values, Dict("latitude" => "latitude"), ["lat"]);
    #append_nc!(ncresult, "lon", _lon_values, Dict("longitude" => "longitude"), ["lon"]);
    append_nc!(ncresult, "chl", _mat_chls, Dict("chl" => "chl"), ["lon", "lat"]);
    append_nc!(ncresult, "lai", _mat_lais, Dict("lai" => "lai"), ["lon", "lat"]);
    append_nc!(ncresult, "lma", _mat_cbcs + _mat_pros, Dict("lma" => "lma"), ["lon", "lat"]);
    append_nc!(ncresult, "lwc", _mat_lwcs, Dict("lwc" => "lwc"), ["lon", "lat"]);
    append_nc!(ncresult, "cbc", _mat_cbcs, Dict("cbc" => "cbc"), ["lon", "lat"]);
    append_nc!(ncresult, "pro", _mat_pros, Dict("pro" => "pro"), ["lon", "lat"]);
    #append_nc!(ncresult, "car", _mat_cars, Dict("car" => "car"), ["lon", "lat"]);
    
    GC.gc()

    return _fittings
end;

#
# cd path_of_project
# julia --project
#     include("shift_spectra.jl");
#     fit_shift_traits!(inputdata, outputresult);
#

#=
# An example of how to use the fitting function
function fit_example!()
    _#file = "/net/fluo/data1/students/renato/aviris_dangermond/aviris_dangermond_point_conception_time_00.nc";
    _file = "/net/fluo/data1/students/renato/aviris_dangermond/aviris_dangermond_time_00/output_clipped.nc";
    _wlen = read_nc(FT, _file, "wavelength");
    _refl = read_nc(FT, _file, "reflectance");

    #_ref_xy = (_wlen, _refl[200,:,200,1]);
    _ref_xy = (_wlen, _refl[200,200,:,1]);
    # (_dict_func, _fit_func, _ms, _st) = solver_params(_ref_xy, ["cab", "car", "lai", "lma"]);
    (_dict_func, _fit_func, _ms, _st) = solver_params(_ref_xy, ["cab", "lai", "lma"]);
    _sol = find_peak(_fit_func, _ms, _st);
    # _cur = target_curve(_ref_xy[1], _dict_func(_sol));

    #@show _sol;
    #@show _cur;

    #(_dict_func, _fit_func, _ms, _st) = solver_params(_ref_xy, ["cab", "car", "lai", "lma"]; soil_color = 2);
    #_sol = find_peak(_fit_func, _ms, _st);
    #_cur = target_curve(_ref_xy[1], _dict_func(_sol));

    #@show _sol;
    #@show _cur;

    return nothing
end;

# Function to try the data
function test_fit!()
    #_file = "/net/fluo/data1/students/renato/aviris_dangermond/aviris_dangermond_point_conception_time_00.nc";
    _file = "/net/fluo/data1/students/renato/aviris_dangermond/aviris_dangermond_time_00/output_clipped.nc";
    _wlen = read_nc(FT, _file, "wavelength");
    _refl = read_nc(FT, _file, "reflectance");

    # fit the curve for one data
    #_mask_veg = (_refl[:,120,:,1] .> 0.05) .&& (_refl[:,220,:,1] .> 0.05) .&& (_refl[:,320,:,1] .> 0.05);
     _mask_veg = (_refl[:,:,120,1] .> 0.05) .&& (_refl[:,:,220,1] .> 0.05) .&& (_refl[:,:,320,1] .> 0.05);

    @inline fit_order(i::Int, j::Int, plotting::Bool = false; counts::Union{Int,Nothing} = nothing) = (
        if !(_mask_veg[i,j])
            return nothing
        end;

        # find the optimal fitting order
        #_ref_xy = (_wlen, _refl[i,:,j,1]);
        _ref_xy = (_wlen, _refl[i,j,:,1]);
        _curves = [];
        _labels = [];
        _all_vars = ["cab", "car", "cbc", "ci", "lai", "pro"];
        _fit_order = [];
        while length(_fit_order) < (isnothing(counts) ? length(_all_vars) : counts)
            _new_vars = _all_vars[[!(_fit_var in _fit_order) for _fit_var in _all_vars]];
            _vars_list = [String[_fit_order; _var] for _var in _new_vars];
            _rmses = [];
            _sols = [];
            _temp_curves = [];
            for _vars in _vars_list
                (_dict_func, _fit_func, _ms, _st) = solver_params(_ref_xy, _vars);
                _sol = find_peak(_fit_func, _ms, _st);
                _cur = target_curve(_ref_xy[1], _dict_func(_sol));
                push!(_rmses, _fit_func(_sol));
                push!(_sols, _sol);
                push!(_temp_curves, _cur);
            end;
            _opt_ind = findmax(_rmses)[2];
            push!(_fit_order, _new_vars[_opt_ind]);
            push!(_curves, _temp_curves[_opt_ind]);
            push!(_labels, deepcopy(_fit_order));
            isnothing(counts) ? (@show _fit_order; @show _sols[_opt_ind]) : nothing;
        end;

        # save the fitting results
        if plotting
            _fig = canvas(2; figsize = (12,4));
            _axs = _fig.axes;
            _axs[1].plot(_ref_xy[1], _ref_xy[2], "k-");
            for _i in eachindex(_labels)
                _axs[1].plot(_ref_xy[1], _curves[_i]; label = _labels[_i]);
            end;
            _axs[1].legend();
            save_canvas!(_fig, "test"; formats = ["png"]);
        end;

        return _fit_order
    );

    # find the best orders
    # for _i in 25:50:400, _j in 25:50:400
    #     @show _i, _j, fit_order(_i,_j, false; counts = 3);
    # end;

    @inline fit_best_three(i::Int, j::Int) = (
        @show i, j;

        if !(_mask_veg[i,j])
            return FT[NaN, NaN, NaN]
        else
            #_ref_xy = (_wlen, _refl[i,:,j,1]);
            _ref_xy = (_wlen, _refl[i,j,:,1]);
            (_, _fit_func, _ms, _st) = solver_params(_ref_xy, String["cab", "cbc", "lai"]);

            return find_peak(_fit_func, _ms, _st);
        end;
    );

    # fit_order(275, 375, true; counts = nothing);
    # @show fit_best_three(275, 375);
    # return nothing

    # find the best orders
    _is = collect(25:50:400);
    _js = collect(25:50:400);
    _fits = fit_best_three.(_is', _js);
    _cabs = ones(FT, size(_fits));
    _cbcs = ones(FT, size(_fits));
    _lais = ones(FT, size(_fits));
    for _i in eachindex(_is), _j in eachindex(_js)
        (_cabs[_i,_j],_cbcs[_i,_j],_lais[_i,_j]) = _fits[_i,_j];
    end;

    return _fits,_cabs,_cbcs,_lais
end
=#
