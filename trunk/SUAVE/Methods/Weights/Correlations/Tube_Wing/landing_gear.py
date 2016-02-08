# landing_gear.py
# 
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2016, E. Botero

# ----------------------------------------------------------------------
#   Landing Gear
# ----------------------------------------------------------------------

def landing_gear(TOW, landing_gear_wt_factor = 0.04):
    """ weight = SUAVE.Methods.Weights.Correlations.Tube_Wing.landing_gear(TOW)
        Calculate the weight of the landing gear assuming that the gear 
        weight is 4 percent of the takeoff weight        
        
        Inputs:
            TOW - takeoff weight of the aircraft [kilograms]
            landing_gear_wt_factor - landing gear weight as percentage of TOW [dimensionless]
        
        Outputs:
            weight - weight of the landing gear [kilograms]
            
        Assumptions:
            calculating the landing gear weight based on the takeoff weight 
    """ 
    
    #process
    weight = landing_gear_wt_factor * TOW
    
    return weight