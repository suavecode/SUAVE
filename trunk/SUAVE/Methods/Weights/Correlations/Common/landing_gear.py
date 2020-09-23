## @ingroup Methods-Weights-Correlations-Common 
# landing_gear.py
# 
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2016, E. Botero
#           Jul 2017, M. Clarke

# ----------------------------------------------------------------------
#   Landing Gear
# ----------------------------------------------------------------------
from SUAVE.Core import Data


## @ingroup Methods-Weights-Correlations-Common
def landing_gear(vehicle, landing_gear_wt_factor=0.04):
    """ Calculate the weight of the landing gear assuming that the gear 
    weight is 4 percent of the takeoff weight        
    
    Assumptions:
        calculating the landing gear weight based on the takeoff weight
    
    Source: 
        N/A
        
    Inputs:
        TOW - takeoff weight of the aircraft                              [kilograms]
        landing_gear_wt_factor - landing gear weight as percentage of TOW [dimensionless]
    
    Outputs:
        weight - weight of the landing gear                               [kilograms]
            
    Properties Used:
        N/A
    """

    # process
    weight          = landing_gear_wt_factor * vehicle.mass_properties.max_takeoff
    output          = Data()
    output.main     = weight * 0.9
    output.nose     = weight * 0.1
    return output
