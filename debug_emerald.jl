using Emerald
using Emerald.EmeraldData.WeatherDrivers
using Emerald.EmeraldLand.Namespace
using Emerald.EmeraldFrontier: simulation!, SAVING_DICT
using Emerald.EmeraldLand.Namespace: SPACConfiguration
using Emerald.EmeraldData.GlobalDatasets: grid_spac
using DataFrames

# Define Floating Type
FT = Float64

# **1️⃣ Ensure GPP is Saved**
SAVING_DICT["GPP"] = true  # ✅ Explicitly enable GPP saving

# **2️⃣ Setup Fake Configuration**
CONFIG = SPACConfiguration(FT)

# **3️⃣ Create Fake Environmental Data**
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
    "SOIL_COLOR" => FT(10.0),    
    "SAI" => FT(0.2),            

    # ✅ Fix for Soil Parameters (4 Layers)
    "SOIL_N" => [1.37 for _ in 1:4],  
    "SOIL_α" => [163.2656 for _ in 1:4],  
    "SOIL_ΘR" => [0.034 for _ in 1:4],   
    "SOIL_ΘS" => [0.46 for _ in 1:4],    
    "SOIL_Ks" => [0.01 for _ in 1:4],    
    "SOIL_Ψs" => [-0.5 for _ in 1:4],    

    # ✅ Soil and Leaf Temperature should match the 4 layers
    "T_LEAF" => FT(298.15),       
    "T_SOIL_1" => FT(295.15),    
    "T_SOIL_2" => FT(294.15),    
    "T_SOIL_3" => FT(293.15),    
    "T_SOIL_4" => FT(292.15),    
    "SWC_1" => FT(0.3),          
    "SWC_2" => FT(0.25),         
    "SWC_3" => FT(0.2),          
    "SWC_4" => FT(0.15)          
) |> Dict{String, Any}

# **4️⃣ Initialize Fake SPAC**
spac_fake = grid_spac(CONFIG, dict_fake)

# **5️⃣ Generate Fake Weather Data (Multiple Time Steps)**
fake_weather_df = DataFrame(
    FDOY = 190:200,  # Multiple days
    RAD = rand(11) * 500,  
    T_AIR = rand(11) * 10 .+ 280,  
    SWC_1 = rand(11) * 0.3,  
    PRECIP = rand(11) * 0.01  
)

# ✅ Add Missing Fields Dynamically
missing_columns = Dict(
    :CLOUD => 0.3,
    :T_SOIL_1 => 288.0,
    :VCMAX25 => 60.0,
    :JMAX25 => 100.0,
    :CHLOROPHYLL => 40.0,
    :LAI => 3.0,
    :VPD => rand(11) * 1.5  # ✅ FIXED: Ensure VPD exists!
)

for (col, value) in missing_columns
    if !(col in names(fake_weather_df))
        fake_weather_df[!, col] .= value
    end
end

# **6️⃣ Debug: Check Initial Conditions**
println("\n🔍 Initial Conditions:")
println("🔹 Solar Radiation (RAD_DIF + RAD_DIR): ", fake_weather_df.RAD)
println("🔹 VPD: ", fake_weather_df.VPD)
println("🔹 Air Temperature (T_AIR): ", fake_weather_df.T_AIR)
println("🔹 Soil Water Content (SWC_1): ", fake_weather_df.SWC_1)

# **7️⃣ Run Simulation**
println("\n🚀 Running Minimal Simulation with Fake Data...")
try
    simulation!(CONFIG, spac_fake, fake_weather_df; saving_dict=SAVING_DICT)
    println("✅ simulation! ran successfully!")
catch e
    println("🚨 Error in simulation!: ", e)
end

# **8️⃣ Debug: Check Simulation Outputs (Hide Zeros)**
df_result = DataFrame(fake_weather_df)

println("\n🔍 Checking Simulation Outputs...")
println("🔹 Available Columns: ", names(df_result))

if :GPP in names(df_result)
    nonzero_gpp = filter(row -> row.GPP != 0, eachrow(df_result))
    
    if isempty(nonzero_gpp)
        println("🚨 ERROR: `GPP` is present but all values are zero!")
    else
        println("✅ `GPP` Output (Filtered Non-Zero): ")
        for row in nonzero_gpp
            println("📆 FDOY $(row.FDOY) → GPP: ", row.GPP)
        end
    end
else
    println("🚨 ERROR: `GPP` is missing from output!")
end

# **9️⃣ Debug: Verify GPP Computation Per Time Step (Hide Zeros)**
println("\n🔍 Verifying GPP Computation Per Time Step:")
for row in eachrow(df_result)
    gpp_value = hasproperty(row, :GPP) ? (row.GPP != 0 ? row.GPP : "N/A") : "N/A"
    println("📆 FDOY $(row.FDOY) → GPP: ", gpp_value)
end
