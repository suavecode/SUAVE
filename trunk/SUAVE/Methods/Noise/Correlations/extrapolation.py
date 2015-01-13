# extrapolation.py
# 
# Created:  Andrew Wendorff, July 2014
# Modified:         


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Attributes import Units as Units
from SUAVE.Core import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#   Method
# ----------------------------------------------------------------------

def extrapolation(weight_landing, number_of_engines, thrust_sea_level, thrust_landing):
    """ weight = SUAVE.Methods.Weights.Correlations.Tube_Wing.landing_gear(TOW)
                
        
        Inputs:
            
        
        Outputs:
            
            
        Assumptions:
        The baseline case is 101 PNdb, 25,000 lb thrust, 1 engine, 1000ft
             
    """ 
    
    #process
    baseline_noise  = 101 
    thrust_percentage = (thrust_sea_level/ Units.force_pound)/25000 * 100.
    thrust_reduction  = thrust_landing/thrust_sea_level * 100.
    noise_increase_due_to_thrust = -0.0002193 * thrust_percentage ** 2. + 0.09454 * thrust_percentage - 7.30116 
    noise_landing = - 0.0015766 * thrust_reduction ** 2. + 0.34882 * thrust_reduction -19.2569
    takeoff_distance_noise = -4. # 1500 ft altitude at 6500m from start of take-off
    sideline_distance_noise = -6.5 #1 476 ft (450m) from centerline (effective distance = 1476*1.25 = 1845ft)
    landing_distance_noise = 9.1 # 370 ft altitude at 6562 ft (2000m) from runway    
    
    takeoff   = 10. * np.log10(10. ** (baseline_noise/10.) * number_of_engines) - 4. + takeoff_distance_noise + noise_increase_due_to_thrust
    side_line = 10. * np.log10(10. ** (baseline_noise/10.) * number_of_engines) - 4. + sideline_distance_noise + noise_increase_due_to_thrust
    landing   = 10. * np.log10(10. ** (baseline_noise/10.) * number_of_engines) - 5. + landing_distance_noise + noise_increase_due_to_thrust + noise_landing
    airframe  = 40. + 10. * np.log10(weight_landing)
    
    output = Data()
    output.takeoff   = takeoff
    output.side_line = side_line
    output.landing   = 10. * np.log10(10. ** (airframe/10.) + 10. ** (landing/10.))
    
    return output