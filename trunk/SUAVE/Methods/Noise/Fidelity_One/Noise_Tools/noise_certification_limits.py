# noise_certification_limits.py
# 
# Created:  Jul 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
#   Noise Certification Limits
# ----------------------------------------------------------------------

def noise_certification_limits(results,vehicle):
    """ SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.noise_certification_limits(results,vehicle):
            Computes the certification noise limits as a function of the aircraft weight [lbs] and number of engines for each segment.

            Inputs:
                vehicle	 - SUAVE type vehicle
                results

            Outputs: Noise limits in EPNL
                noise_approach_limit             - Approach noise limit as a function of the landing weight, [EPNdB]
                noise_flyover_limit              - Flyover noise limit as a function of the takeoff weight, [EPNdB]
                noise_sideline_limit             - Sideline noise limit as a function of the takeoff weight, [EPNdB]

            Assumptions:
                None."""
    
    #unpack
    weight_approach     = np.float(results.approach.segments.descent.conditions.weights.total_mass[-1]) / Units.lbs
    weight_tow_mission  = np.float(results.flyover.segments.climb.conditions.weights.total_mass[-1])     / Units.lbs
    n_engines           = np.int(vehicle.propulsors.turbofan.number_of_engines)
    
    #Determination of the number of engines
    if n_engines > 3:
        C_flyover = 8.96*(10**-3.)
    elif n_engines == 3:
        C_flyover = 1.27*(10**-2.)
    else:
        C_flyover = 2.13*(10**-2)
    
    #Constants for the Stage III noise limits
    T_flyover  = 4.
    C_approach = 1.68*(10**-8.)
    T_approach = 2.33
    C_sideline = 6.82*(10**-7.)
    T_sideline = 2.56
    
    #Calculation of noise limits based on the weight
    noise_sideline_limit = np.around(np.log((weight_tow_mission/C_sideline))* T_sideline /np.log(2),decimals=1)
    noise_flyover_limit  = np.around(np.log((weight_tow_mission/C_flyover)) * T_flyover  /np.log(2),decimals=1)
    noise_approach_limit = np.around(np.log((weight_approach   /C_approach))* T_approach /np.log(2),decimals=1)

    return (noise_approach_limit,noise_flyover_limit,noise_sideline_limit)

def noise_certification_propeller (noise_data):
    """ SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.noise_certification_propeller(results,vehicle):
                Computes the certification noise limit as a function of the aircraft weight [lbs] in dbA for a Propeller driven aircraft.
    
                Inputs:
                    noise_data
    
                Outputs: Noise limits in db(A)
                    noise_takeoff_limit             - Takeoff noise limit as a function of the takeoff weight, [dbA]

                Assumptions:
                    None."""
    
    #unpack
    weight_tow_mission = noise_data.tow_weight / Units.lbs 
    
    #Calculation of noise limit based on the aircraft weight - FAA AC36-1H Appendix 7
    if weight_tow_mission <= 1320.0:
        noise_takeoff_limit = 68.00
    elif weight_tow_mission <= 3300.0:
        noise_takeoff_limit = 68.00 + (weight_tow_mission - 1320.0)/165 
    else:
        noise_takeoff_limit = 80.00 
        
    return (noise_takeoff_limit)
    
    
