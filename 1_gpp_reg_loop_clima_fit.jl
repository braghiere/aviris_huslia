using NetcdfIO: append_nc!, create_nc!, read_nc
using Distributed: @everywhere, addprocs, pmap, workers, rmprocs
using ProgressMeter: @showprogress

using Emerald.EmeraldData.GlobalDatasets: LandDatasetLabels, grid_dict, grid_spac
using Emerald.EmeraldData.WeatherDrivers: grid_weather_driver
using Emerald.EmeraldLand.Namespace: SPACConfiguration
using Emerald.EmeraldLand.Namespace: BetaFunction, BetaParameterG1, BetaParameterPsoil, BetaParameterΘ, MedlynSM

using Dates
using NetCDF
using Base.GC


# Set up the global variables
FT = Float64;
#
# Define a configuration for the simulation
#
@info "Define a configuration for the simulation...";
CONFIG = SPACConfiguration(Float64);

#
# You can and need to replace the traits in the dict to match the observations, you function update! for more information
# Here I use year 2020 because the LAI data is not available for 2022 (but it will be rewritten anyway)
#
@info "Reading the traits data using GriddingMachine...";
#dict_shift = gm_dict(GriddingMachineLabels(year = 2020), 34.448598, -120.471551);
dict_shift = grid_dict(LandDatasetLabels("gm2", 2020), 34.448598, -117.471551);
dict_shift["LONGITUDE"] = -120.471551;
dict_shift["LMA"] = 0.01;
dict_shift["soil_color"] = 13;
dict_shift["SOIL_N"] = [1.37 for _ in 1:4];
dict_shift["SOIL_α"] = [163.2656 for _ in 1:4];
dict_shift["SOIL_ΘR"] = [0.034 for _ in 1:4];
dict_shift["SOIL_ΘS"] = [0.46 for _ in 1:4];


#
# Define a default SPAC, will be copied to each thread
#
@info "Define a default SPAC to work on...";
spac_shift = grid_spac(CONFIG, dict_shift);
@everywhere linear_θ_soil(x) = min(1, max(eps(), (x - 0.034) / (0.46 - 0.034)));
g1 = dict_shift["G1_MEDLYN_C3"];
bt = BetaFunction{Float64}(FUNC = linear_θ_soil, PARAM_X = BetaParameterΘ(), PARAM_Y = BetaParameterG1());
for leaf in spac_shift.plant.leaves
    leaf.flux.trait.stomatal_model = MedlynSM{Float64}(G0 = 0.005, G1 = g1, β = bt);
end;


#
# Read the weather driver data
# TODO: replace 2020 with 2022 once the data is regridded
#
@info "Preparing the weather driver data...";
dict_shift["YEAR"] = 2022;
wdrv_shift = grid_weather_driver("wd1", dict_shift);
wdrv_shift.PRECIP .= 0;


#
# And you can customize the number of threads here
#
@info "Create workers to run the simulation...";
if length(workers()) == 1
    addprocs(32; exeflags = "--project");
end;
@everywhere include("1_pmap.jl");
@everywhere linear_θ_soil(x) = min(1, max(eps(), (x - 0.034) / (0.46 - 0.034)));

#
# Function to run the simulation for one day or one hour
# To use this function you need to provide
#     - which day to start based on the day of year
#     - the chl array
#     - the lai array
#     - the vcmax25 array
#
function run_one_timestep!(day::Int, chls::Matrix, lais::Matrix, vcms::Matrix)
#function run_one_timestep!(day::Int, chls::FT, lais::FT, vcms::FT)
    @assert length(chls) == length(lais) == length(vcms) "Size of chls, lais, and lmas must be the same!";

    n = findfirst(wdrv_shift.FDOY .> day .&& wdrv_shift.RAD .> 1);
    oneday_df = wdrv_shift[n:n+23,:];
    _,m = findmax(oneday_df.RAD);
    onehour_df = oneday_df[m:m,:];

    # prepare the data used for paralleled simulation
    params = [];
    for i in eachindex(chls)
        if any(isnan, (chls[i], lais[i], vcms[i]))
            push!(params, [nothing, nothing, nothing]);
        else
            df = deepcopy(onehour_df);
            df.CHLOROPHYLL .= chls[i];
            df.car .= chls[i]/7.;
            df.LAI .= lais[i];
            #From Croft et al. (2017)
            df.VCMAX25 .= 1.30.*chls[i] .+ 3.72;
            df.JMAX25 .= 2.49.*chls[i] .+ 10.80;
            df.LMA .= vcms[i];
            #param = [CONFIG, deepcopy(spac_shift), df];
            param = [CONFIG, spac_shift, df];
            push!(params, param);
        end;
    end;

    #@show Base.summarysize(params) / length(params);

    # use pmap to run the simulation in parallel
    results = @showprogress pmap(run_shift_simulation!, params);
    gpps = [r[1] for r in results];
    sifs740 = [r[2] for r in results];
    sifs683 = [r[3] for r in results];
    sifs757 = [r[4] for r in results];
    sifs771 = [r[5] for r in results];
    transps = [r[6] for r in results];
    
    oneday_df = nothing;
    onehour_df = nothing;
    params = nothing;
    results = nothing;
    
    GC.gc()

    return reshape(gpps, size(chls,1), size(chls,2)), reshape(sifs740, size(chls,1), size(chls,2)), reshape(sifs683, size(chls,1), size(chls,2)), reshape(sifs757, size(chls,1), size(chls,2)), reshape(sifs771, size(chls,1), size(chls,2)), reshape(transps, size(chls,1), size(chls,2))
end;


#
# Here is an example of how to use the function
#
@info "Run the example...";
# Given dates
#
dates = ["2022-02-24T00:00:00.000000", "2022-02-28T00:00:00.000000", "2022-03-08T00:00:00.000000", "2022-03-16T00:00:00.000000","2022-03-22T00:00:00.000000", "2022-04-05T00:00:00.000000", "2022-04-12T00:00:00.000000", "2022-04-20T00:00:00.000000","2022-04-29T00:00:00.000000", "2022-05-03T00:00:00.000000", "2022-05-11T00:00:00.000000", "2022-05-17T00:00:00.000000","2022-05-29T00:00:00.000000"]

# Loop through dates and calculate day of the year
for (day_index, date) in enumerate(dates)


    # Calculate day of the year
    day_of_year = Dates.dayofyear(Dates.DateTime(date, "yyyy-mm-ddTHH:MM:SS.ssssss"))
    @info "Processing data for day $day_of_year (Date: $date)..."

    # Format the day index with leading zeros
    day_str = lpad(day_index - 1, 2, '0')
    
    # Generate the file name with leading zeros and correct day index
    chl_file = "/net/squid/data3/data/FluoData1/students/renato/aviris_dangermond/traits/datasets/clima_fit/chl_aviris_dangermond_clima_fit_time_$(day_str)_reg.nc"
    lai_file = "/net/squid/data3/data/FluoData1/students/renato/aviris_dangermond/traits/datasets/clima_fit/lai_aviris_dangermond_clima_fit_time_$(day_str)_reg.nc"
    vcm_file = "/net/squid/data3/data/FluoData1/students/renato/aviris_dangermond/traits/datasets/clima_fit/lma_aviris_dangermond_clima_fit_time_$(day_str)_reg.nc"


    chls_1 = read_nc(FT, chl_file, "chl")
    lais_1 = read_nc(FT, lai_file, "lai")
    vcms_1 = read_nc(FT, vcm_file, "lma")

    # Extract the data arrays from the files (adjust the indices accordingly)
    chls_data = chls_1[1:492,1:458,1]
    lais_data = lais_1[1:492,1:458,1]
    vcms_data = vcms_1[1:492,1:458,1]
    #chls_data = chls_1[201:210,201:210,1]
    #lais_data = lais_1[201:210,201:210,1]
    #vcms_data = vcms_1[201:210,201:210,1]

    # Run the simulation for the current day
    gpps, sifs740, sifs683, sifs757, sifs771, transps = run_one_timestep!(day_of_year, chls_data, lais_data, vcms_data)

    # Save the results to a netCDF file
    ncresult = "shift_fluxes_day_$(day_str)_clima_fit_reg_jmax.nc"
    #lats_1 = read_nc(FT, chls_file, "lat")[1:size(chls_data, 2)]
    #lons_1 = read_nc(FT, chls_file, "lon")[1:size(chls_data, 1)]

    lats_1 = read_nc(FT, chl_file, "lat");
    lats = lats_1[1:458];
    #lats = lats_1[201:210];

    lons_1 = read_nc(FT, chl_file, "lon");
    lons = lons_1[1:492];
    #lons = lons_1[201:210];
    
    #time_1 = read_nc(DateTime, chl_file, "time");
    #times = time_1[1];
    
    # Convert DateTime values to numeric representation (seconds since a specific date) using BigInt
    start_date = Dates.DateTime(date, "yyyy-mm-ddTHH:MM:SS.ssssss")
    
    # Convert start_date to a UNIX timestamp (seconds since January 1, 1970)
    start_date_unix = Dates.value(start_date)

    # Create an array with the start_date_unix value
    time_values = [start_date_unix]


    
    #_len_lat = size(vcms_1, 2);
    #_len_lon = size(vcms_1, 1);
    _len_time = 1; # size(_refl, 1);
    _len_lat = 458; # size(_refl, 1);
    _len_lon = 492; # size(_refl, 3);
    #_len_lat = 10; # size(_refl, 1);
    #_len_lon = 10; # size(_refl, 3);
    # save the matrices to a file
    
    gpps_reshaped = reshape(gpps, (_len_lon, _len_lat, _len_time))
    sifs740_reshaped = reshape(sifs740, (_len_lon, _len_lat, _len_time))
    sifs683_reshaped = reshape(sifs683, (_len_lon, _len_lat, _len_time))
    sifs757_reshaped = reshape(sifs757, (_len_lon, _len_lat, _len_time))
    sifs771_reshaped = reshape(sifs771, (_len_lon, _len_lat, _len_time))
    transps_reshaped = reshape(transps, (_len_lon, _len_lat, _len_time)) 
    
    create_nc!(ncresult, String["time", "lat", "lon"], [_len_time, _len_lat, _len_lon]);
    #create_nc!(ncresult, String["lat", "lon"], [size(chls_data, 2), size(chls_data, 1)])
    time_attributes = Dict("standard_name" => "time",
                       "units" => "seconds since $date",
                       "calendar" => "gregorian",
                       "axis" => "T")
    append_nc!(ncresult, "time", time_values, time_attributes, ["time"])
    # Define attributes (for example, standard_name and units)
    append_nc!(ncresult, "lat", lats, Dict("latitude" => "latitude"), ["lat"])
    append_nc!(ncresult, "lon", lons, Dict("longitude" => "longitude"), ["lon"])
    append_nc!(ncresult, "gpp", gpps_reshaped, Dict("gpp" => "gpp"), ["lon", "lat", "time"])
    append_nc!(ncresult, "sif740", sifs740_reshaped, Dict("sif740" => "sif740"), ["lon", "lat", "time"])
    append_nc!(ncresult, "sif683", sifs683_reshaped, Dict("sif683" => "sif683"), ["lon", "lat", "time"])
    append_nc!(ncresult, "sif757", sifs757_reshaped, Dict("sif757" => "sif757"), ["lon", "lat", "time"])
    append_nc!(ncresult, "sif771", sifs771_reshaped, Dict("sif771" => "sif771"), ["lon", "lat", "time"])
    append_nc!(ncresult, "transp", transps_reshaped, Dict("transp" => "transp"), ["lon", "lat", "time"])
    
    ncclose(ncresult)

    chls_1 = nothing;
    lais_1 = nothing;
    vcms_1 = nothing;
    chls_data = nothing;
    lais_data = nothing;
    vcms_data = nothing;
    
    gpps = nothing;
    sifs740 = nothing; 
    sifs683 = nothing;
    sifs757 = nothing;
    sifs771 = nothing;
    transps = nothing;
    
    gpps_reshaped = nothing;
    sifs740_reshaped = nothing;
    sifs683_reshaped = nothing;
    sifs757_reshaped = nothing;
    sifs771_reshaped = nothing;
    transps_reshaped = nothing;
    
    lats_1 = nothing;
    lats = nothing;
    
    lons_1 = nothing;
    lons = nothing;
    
    
    @info "Data for day $day_of_year processed and saved."
    
    GC.gc()

    # Introduce a one-minute delay between each iteration
    sleep(60)  # Sleep for 60 seconds (1 minute)

end

@info "Processing for all days completed!"
