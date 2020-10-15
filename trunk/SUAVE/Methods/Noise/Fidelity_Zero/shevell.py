## @ingroupMethods-Noise-Fidelity_Zero
# shevell.py
# 
# Created:  Jul 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Units, Data

# ----------------------------------------------------------------------
#   Shevell Method
# ----------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_Zero
def shevell(weight_landing, number_of_engines, thrust_sea_level, thrust_landing):
    """ This uses correlations from Shevell, also used in AA241A/B, to calculate the sources for noise

    Assumptions:
    None

    Source:
    Stanford AA 241A/B Notes : http://aerodesign.stanford.edu/aircraftdesign/aircraftdesign.html

    Inputs:
        weight_landing     [newtons]
        number_of_engines  [int]
        thrust_sea_level   [newtons]
        thrust_landing     [newtons]
    
    Outputs:
        output.
            takeoff        [float]
            side_line      [float]
            landing        [float]

    Properties Used:
        baseline noise = 101.
        various tuned correlations
    """         
    
    

    # process
    baseline_noise    = 101. 
    thrust_percentage = (thrust_sea_level/ Units.force_pound)/25000 * 100.
    thrust_reduction  = thrust_landing/thrust_sea_level * 100.
    
    noise_increase_due_to_thrust = - 0.0002193 * thrust_percentage ** 2. + 0.09454 * thrust_percentage - 7.30116 
    noise_landing                = - 0.0015766 * thrust_reduction ** 2. + 0.34882 * thrust_reduction -19.2569
    
    takeoff_distance_noise  = -4.  # 1500 ft altitude at 6500m from start of take-off
    sideline_distance_noise = -6.5 # 1476 ft (450m) from centerline (effective distance = 1476*1.25 = 1845ft)
    landing_distance_noise  = 9.1  # 370 ft altitude at 6562 ft (2000m) from runway    
    
    takeoff   = 10. * np.log10(10. ** (baseline_noise/10.) * number_of_engines) - 4. \
        + takeoff_distance_noise + noise_increase_due_to_thrust
    side_line = 10. * np.log10(10. ** (baseline_noise/10.) * number_of_engines) - 4. \
        + sideline_distance_noise + noise_increase_due_to_thrust
    landing   = 10. * np.log10(10. ** (baseline_noise/10.) * number_of_engines) - 5. \
        + landing_distance_noise + noise_increase_due_to_thrust + noise_landing
    airframe  = 40. + 10. * np.log10(weight_landing / Units.lbs)
    
    output = Data()
    output.takeoff   = takeoff
    output.side_line = side_line
    output.landing   = 10. * np.log10(10. ** (airframe/10.) + 10. ** (landing/10.))
    
    return output