# shevell.py
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

def shevell(weight_landing, number_of_engines, thrust_sea_level, thrust_landing):
    """ weight = SUAVE.Methods.Noise.Correlations.shevell(weight_landing, number_of_engines, thrust_sea_level, thrust_landing)
                
        Inputs: 
            weight_landing - Landing weight of the aircraft [kilograms]
            number of engines - Number of engines installed on the aircraft [Dimensionless]
            thrust_sea_level - Sea Level Thrust of the Engine [Newtons]
            thrust_landing - Thrust of the aircraft coming in for landing [Newtons]
        
        Outputs:
            output() - Data Class
                takeoff - Noise of the aircraft at takeoff directly along the runway centerline  (500 ft altitude at 6500m from start of take-off) [dB]
                side_line - Noise of the aircraft at takeoff at the sideline measurement station (1,476 ft (450m) from centerline (effective distance = 1476*1.25 = 1845ft)[dB]
                landing - Noise of the aircraft at landing directly along the trajectory (370 ft altitude at 6562 ft (2000m) from runway) [dB]
            
        Assumptions:
        The baseline case used is 101 PNdb, 25,000 lb thrust, 1 engine, 1000ft
        The noise_increase_due_to_thrust and noise_landing are equation extracted from images. 
        This is only meant to give a rough estimate compared to a DC-10 aircraft. As the aircraft configuration varies from this configuration, the validity of the method will also degrade.     
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
    airframe  = 40. + 10. * np.log10(weight_landing / Units.lbs)
    
    output = Data()
    output.takeoff   = takeoff
    output.side_line = side_line
    output.landing   = 10. * np.log10(10. ** (airframe/10.) + 10. ** (landing/10.))
    
    return output