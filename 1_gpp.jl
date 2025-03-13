using NetcdfIO: append_nc!, create_nc!, read_nc
using Distributed: @everywhere, addprocs, pmap, workers
using ProgressMeter: @showprogress

using Emerald.EmeraldFrontier: GriddingMachineLabels, gm_dict, spac, weather_driver
using Emerald.EmeraldLand.Namespace: SPACConfiguration
using Emerald.EmeraldLand.Namespace: BetaFunction, BetaParameterG1, BetaParameterPsoil, MedlynSM

# Set up the global variables
FT = Float64;
#
# Define a configuration for the simulation
#
@info "Define a configuration for the simulation...";
CONFIG = SPACConfiguration{Float64}(DEBUG = true, ENABLE_ENERGY_BUDGET = true, ENABLE_PLANT_HYDRAULICS = true, ENABLE_SOIL_WATER_BUDGET = true, ENABLE_SOIL_EVAPORATION = true);


#
# You can and need to replace the traits in the dict to match the observations, you function update! for more information
# Here I use year 2020 because the LAI data is not available for 2022 (but it will be rewritten anyway)
#
@info "Reading the traits data using GriddingMachine...";
dict_shift = gm_dict(GriddingMachineLabels(year = 2020), 34.448598, -120.471551);
dict_shift["LMA"] = 0.01;
dict_shift["SOIL_N"] = [1.37 for _ in 1:4];
dict_shift["SOIL_α"] = [163.2656 for _ in 1:4];
dict_shift["SOIL_ΘR"] = [0.034 for _ in 1:4];
dict_shift["SOIL_ΘS"] = [0.46 for _ in 1:4];


#
# Define a default SPAC, will be copied to each thread
#
@info "Define a default SPAC to work on...";
spac_shift = spac(dict_shift, CONFIG);
@everywhere linear_p_soil(x) = min(1, max(eps(), 1 + x / 5));
g1 = dict_shift["MEDLYN_G1"];
bt = BetaFunction{Float64}(FUNC = linear_p_soil, PARAM_X = BetaParameterPsoil(), PARAM_Y = BetaParameterG1());
for leaf in spac_shift.LEAVES
    leaf.SM = MedlynSM{Float64}(G0 = 0.005, G1 = g1, β = bt);
end;


#
# Read the weather driver data
# TODO: replace 2020 with 2022 once the data is regridded
#
@info "Preparing the weather driver data...";
dict_shift["YEAR"] = 2022;
wdrv_shift = weather_driver("wd1", dict_shift);
wdrv_shift.PRECIP .= 0;


#
# And you can customize the number of threads here
#
@info "Create workers to run the simulation...";
if length(workers()) == 1
#    addprocs(8; exeflags = "--project");
    addprocs(64; exeflags = "--project");
end;
@everywhere include("1_pmap.jl");
@everywhere linear_p_soil(x) = min(1, max(eps(), 1 + x / 5));


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
            df.LAI .= lais[i];
            df.VCMAX25 .= 1.30.*chls[i] .+ 3.72;
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

    return reshape(gpps, size(chls,1), size(chls,2)), reshape(sifs740, size(chls,1), size(chls,2)), reshape(sifs683, size(chls,1), size(chls,2)), reshape(sifs757, size(chls,1), size(chls,2)), reshape(sifs771, size(chls,1), size(chls,2)), reshape(transps, size(chls,1), size(chls,2))
end;


#
# Here is an example of how to use the function
#
@info "Run the example...";
day = 55;
#chls = 20 .+ rand(12,12) .* 30;

#Read netcdf
file_chl = "/net/fluo/data1/students/renato/aviris_dangermond/traits/chl_aviris_dangermond_time_00.nc";
chls_1 = read_nc(FT, file_chl, "chl");
#chls = chls_1[1600:1611,1600:1611,1];
chls = chls_1[1:3200,1:3200,1];

#lais = 0.2 .+ rand(12,12) .* 3;
file_lai = "/net/fluo/data1/students/renato/aviris_dangermond/traits/lai_aviris_dangermond_time_00.nc";
lais_1 = read_nc(FT, file_lai, "lai");
#lais = lais_1[1600:1611,1600:1611,1];
lais = lais_1[1:3200,1:3200,1];

#vcms = 20 .+ rand(12,12) .* 30;

file_vcm = "/net/fluo/data1/students/renato/aviris_dangermond/traits/lma_aviris_dangermond_time_00.nc";
vcms_1 = read_nc(FT, file_vcm, "lma");
#vcms = vcms_1[1600:1611,1600:1611,1];
vcms = vcms_1[1:3200,1:3200,1];

#chls[1] = NaN;
gpps,sifs740,sifs683,sifs757,sifs771,transps = run_one_timestep!(day, chls, lais, vcms);

ncresult::String = "shift_fluxes_day_00.nc"

#_len_lat = 12; # size(_refl, 1);
#_len_lon = 12; # size(_refl, 3);
_len_lat = 3200; # size(_refl, 1);
_len_lon = 3200; # size(_refl, 3);
_mat_gpps = reshape(gpps, _len_lat, _len_lon);
_mat_sifs740 = reshape(sifs740, _len_lat, _len_lon);
_mat_sifs683 = reshape(sifs683, _len_lat, _len_lon);
_mat_sifs757 = reshape(sifs757, _len_lat, _len_lon);
_mat_sifs771 = reshape(sifs771, _len_lat, _len_lon);
_mat_transps = reshape(transps, _len_lat, _len_lon);

lats_1 = read_nc(FT, file_vcm, "latitude");
#lats = lats_1[1600:1611];
lats = lats_1[1:3200];
lons_1 = read_nc(FT, file_vcm, "longitude");
#lons = lons_1[1600:1611];
lons = lons_1[1:3200];


# save the matrices to a file
create_nc!(ncresult, String["lat", "lon"], [_len_lat, _len_lon]);
append_nc!(ncresult, "lat", lats, Dict("latitude" => "latitude"), ["lat"]);
append_nc!(ncresult, "lon", lons, Dict("longitude" => "longitude"), ["lon"]);
append_nc!(ncresult, "gpp", _mat_gpps, Dict("gpp" => "gpp"), ["lat", "lon"]);
append_nc!(ncresult, "sif740", _mat_sifs740, Dict("sif740" => "sif740"), ["lat", "lon"]);
append_nc!(ncresult, "sif683", _mat_sifs683, Dict("sif683" => "sif683"), ["lat", "lon"]);
append_nc!(ncresult, "sif757", _mat_sifs757, Dict("sif757" => "sif757"), ["lat", "lon"]);
append_nc!(ncresult, "sif771", _mat_sifs771, Dict("sif771" => "sif771"), ["lat", "lon"]);
append_nc!(ncresult, "transp", _mat_transps, Dict("transp" => "transp"), ["lat", "lon"]);

