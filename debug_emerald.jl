using Emerald
using Emerald.EmeraldData.WeatherDrivers
using Emerald.EmeraldLand.Namespace
using Emerald.EmeraldFrontier: simulation!, SAVING_DICT
using Emerald.EmeraldLand.Namespace: SPACConfiguration
using Emerald.EmeraldData.GlobalDatasets: grid_spac
using DataFrames
using CSV  # ✅ CSV handling for export

# =============================================
# ✅ LOGGING SETUP
# =============================================
logfile = open("simulation_log.txt", "w")

function log(msg)
    println(msg)
    println(logfile, msg)
end

log("🟢 Starting Emerald Debug Simulation\n")

# =============================================
# ✅ CONFIGURE
# =============================================
FT = Float64

log("⚙️  Configuring SPAC and Saving Dict...")

# Enable outputs needed for the simulation!
SAVING_DICT["GPP"] = true
SAVING_DICT["CNPP"] = true
SAVING_DICT["MOD_SWC"] = true
SAVING_DICT["MOD_T_SOIL"] = true
SAVING_DICT["MOD_T_MMM"] = true
SAVING_DICT["MOD_P_MMM"] = true   # ✅ Added MOD_P_MMM explicitly
SAVING_DICT["ET_VEG"] = true
SAVING_DICT["K_PLANT"] = true
SAVING_DICT["K_ROOT_STEM"] = true
SAVING_DICT["SIF740"] = true  # Example additional output

CONFIG = SPACConfiguration(FT)

# =============================================
# ✅ DEFINE FAKE ENVIRONMENTAL PARAMETERS
# =============================================
log("🌎 Defining Fake Site Environmental Parameters...")

dict_fake = Dict(
    "LONGITUDE" => -156.4,
    "LATITUDE" => 65.7,
    "LMA" => FT(0.01),
    "soil_color" => Int(13),
    "CANOPY_HEIGHT" => FT(10.0),
    "ELEVATION" => FT(200.0),
    "P_ATM" => FT(101325.0),
    "CHL" => FT(30.0),
    "CI" => FT(1.0),
    "CO2" => FT(400.0),
    "RAD_DIF" => FT(50.0),
    "RAD_DIR" => FT(500.0),
    "FDOY" => FT(200.0),
    "LAI" => FT(2.5),
    "RAD_LW" => FT(300.0),
    "PRECIP" => FT(0.0),
    "T_AIR" => FT(298.15),
    "VCMAX25" => FT(60.0),
    "VPD" => FT(1.2),
    "WIND" => FT(2.0),
    "SAI" => FT(0.2),
    "SOIL_COLOR" => FT(10.0),
    "SOIL_N" => [1.37 for _ in 1:4],
    "SOIL_α" => [163.2656 for _ in 1:4],
    "SOIL_ΘR" => [0.034 for _ in 1:4],
    "SOIL_ΘS" => [0.46 for _ in 1:4],
    "SOIL_Ks" => [0.01 for _ in 1:4],
    "SOIL_Ψs" => [-0.5 for _ in 1:4],
    "T_LEAF" => FT(298.15),
    "T_SOIL_1" => FT(295.15),
    "T_SOIL_2" => FT(294.15),
    "T_SOIL_3" => FT(293.15),
    "T_SOIL_4" => FT(292.15),
    "SWC_1" => FT(0.3),
    "SWC_2" => FT(0.25),
    "SWC_3" => FT(0.2),
    "SWC_4" => FT(0.15)
)

spac_fake = try
    grid_spac(CONFIG, dict_fake)
catch e
    log("🚨 Error creating spac_fake: $e")
    close(logfile)
    error("Terminating script due to failed SPAC initialization!")
end

if spac_fake === nothing
    log("🚨 Cannot continue without a valid spac_fake object!")
    close(logfile)
    error("Terminating script!")
end

# =============================================
# ✅ CREATE FAKE WEATHER DRIVER DATA
# =============================================
log("🌤️  Creating Fake Weather Driver Data...")

fake_weather_df = DataFrame(
    FDOY = 190:200,
    RAD = rand(11) .* 500,
    T_AIR = rand(11) .* 10 .+ 285,
    SWC_1 = rand(11) .* 0.2 .+ 0.2,
    PRECIP = rand(11) .* 0.01
)

# Split RAD into RAD_DIR and RAD_DIF
fake_weather_df[!, :RAD_DIR] = fake_weather_df.RAD .* 0.7
fake_weather_df[!, :RAD_DIF] = fake_weather_df.RAD .* 0.3

# =============================================
# ✅ ADD ALL REQUIRED INPUT FIELDS FOR prescribe!
# =============================================
required_columns = Dict(
    :P_ATM      => 101325.0,
    :VPD        => 1.2,
    :CHL        => 30.0,
    :CI         => 1.0,
    :CO2        => 400.0,
    :RAD_LW     => 300.0,
    :LAI        => 3.0,
    :VCMAX25    => 60.0,
    :JMAX25     => 100.0,
    :CLOUD      => 0.3,
    :WIND       => 2.0,
    :T_SOIL_1   => 288.0,
    :T_SOIL_2   => 287.0,
    :T_SOIL_3   => 286.0,
    :T_SOIL_4   => 285.0,
    :T_LEAF     => 298.0,
    :SWC_2      => 0.25,
    :SWC_3      => 0.20,
    :SWC_4      => 0.15
)

for (col, value) in required_columns
    if !(col in names(fake_weather_df))
        fake_weather_df[!, col] .= value
    end
end

# =============================================
# ✅ ADD OUTPUT FIELDS BASED ON SAVING_DICT
# =============================================
n_rows = nrow(fake_weather_df)

for (key, save) in SAVING_DICT
    if save
        if key == "MOD_SWC"
            for i in 1:length(spac_fake.soils)
                col = Symbol("MOD_SWC_$i")
                fake_weather_df[!, col] = fill(NaN, n_rows)
                log("➕ Added column $col")
            end
        elseif key == "MOD_T_SOIL"
            for i in 1:length(spac_fake.soils)
                col = Symbol("MOD_T_SOIL_$i")
                fake_weather_df[!, col] = fill(NaN, n_rows)
                log("➕ Added column $col")
            end
        elseif key == "MOD_T_MMM"
            for label in ["MOD_T_L_MAX", "MOD_T_L_MEAN", "MOD_T_L_MIN"]
                fake_weather_df[!, Symbol(label)] = fill(NaN, n_rows)
                log("➕ Added column $label")
            end
        elseif key == "MOD_P_MMM"
            for label in ["MOD_P_L_MAX", "MOD_P_L_MEAN", "MOD_P_L_MIN"]
                fake_weather_df[!, Symbol(label)] = fill(NaN, n_rows)
                log("➕ Added column $label")
            end
        else
            # General output variables (ET_VEG, GPP, CNPP, etc.)
            col = Symbol(key)
            if !(col in names(fake_weather_df))
                fake_weather_df[!, col] = fill(NaN, n_rows)
                log("➕ Added column $col")
            end
        end
    end
end

log("🧮 Columns present in fake_weather_df: $(names(fake_weather_df))")

# =============================================
# ✅ DEBUG INITIAL CONDITIONS
# =============================================
log("🔍 Initial Conditions:")
log("🔹 RAD_DIR: $(fake_weather_df.RAD_DIR)")
log("🔹 RAD_DIF: $(fake_weather_df.RAD_DIF)")
log("🔹 VPD: $(fake_weather_df.VPD)")
log("🔹 T_AIR: $(fake_weather_df.T_AIR)")
log("🔹 SWC_1: $(fake_weather_df.SWC_1)")

# =============================================
# ✅ RUN SIMULATION AND RETURN RESULTS
# =============================================
log("\n🚀 Running Minimal Simulation with Fake Data...")

wdf = NamedTuple{Tuple(Symbol.(names(fake_weather_df)))}(
    Tuple([fake_weather_df[:, n] for n in names(fake_weather_df)])
)

try
    simulation!(CONFIG, spac_fake, wdf; saving_dict=SAVING_DICT)
    log("✅ simulation! ran successfully!")
catch e
    log("🚨 Error running simulation!: $e")
end

# =============================================
# ✅ CONVERT RESULTS TO DATAFRAME
# =============================================
df_result = DataFrame(wdf)

# =============================================
# ✅ CHECK SIMULATION OUTPUTS
# =============================================
log("\n🔍 Checking Simulation Outputs...")
log("🔹 Available Columns: $(names(df_result))")

if :GPP in names(df_result)
    nonzero_gpp = filter(row -> row.GPP != 0, eachrow(df_result))
    if isempty(nonzero_gpp)
        log("🚨 ERROR: `GPP` is present but all values are zero!")
    else
        log("✅ `GPP` Output (Filtered Non-Zero):")
        for row in nonzero_gpp
            log("📆 FDOY $(row.FDOY) → GPP: $(row.GPP)")
        end
    end
else
    log("🚨 ERROR: `GPP` is missing from output!")
end

# =============================================
# ✅ VERIFY GPP FOR EACH TIME STEP
# =============================================
log("\n🔍 Verifying GPP Computation Per Time Step:")
for row in eachrow(df_result)
    gpp_value = hasproperty(row, :GPP) ? (row.GPP != 0 ? row.GPP : "N/A") : "N/A"
    log("📆 FDOY $(row.FDOY) → GPP: $gpp_value")
end

# =============================================
# ✅ OPTIONAL DEBUGGING: SAVING_DICT + SPAC
# =============================================
log("\n📦 SAVING_DICT: $SAVING_DICT")

try
    sd_field = getfield(spac_fake, :saving_dict)
    log("📦 spac_fake.saving_dict: $sd_field")
catch
    log("⚠️ spac_fake has no saving_dict field (might be okay depending on Emerald version)")
end

# =============================================
# ✅ EXPORT df_result TO CSV
# =============================================
CSV_FILE = "simulation_output.csv"
try
    CSV.write(CSV_FILE, df_result)
    log("✅ Simulation results exported to $CSV_FILE")
catch e
    log("⚠️ Failed to export CSV: $e")
end

log("\n✅ Debug Simulation Complete! Log saved to simulation_log.txt")

# =============================================
# ✅ CLOSE LOG FILE
# =============================================
close(logfile)
