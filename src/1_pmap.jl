using Emerald.EmeraldFrontier: DF_SIMULATIONS, DF_VARIABLES, simulation!
# using Emerald.EmeraldLand.SPAC: update!
using Base.GC


function run_shift_simulation!(param::Vector)
    if any(isnothing, param)
        return NaN, NaN, NaN, NaN, NaN, NaN
    end;

    config = param[1];
    #spac = param[2];
    spac = deepcopy(param[2]);
    df = param[3];

    # add the fields to store outputs
    for label in DF_VARIABLES
        df[!,label] .= 0.0;
    end;
    for label in DF_SIMULATIONS
        df[!,label] .= NaN;
    end;

    simulation!(config, spac, df; initialize_state = true);
    
    spac = nothing;
    
    GC.gc()

    return df.F_GPP[1], df.SIF740[1], df.SIF683[1], df.SIF757[1], df.SIF771[1], df.F_H2O[1]
end;
