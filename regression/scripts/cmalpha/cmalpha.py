# test_cmalpha.py
# Tim Momose, April 2014
# Reference: Aircraft Dynamics: from Modeling to Simulation, by M. R. Napolitano

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cmalpha import taw_cmalpha

import sys
sys.path.append('../Vehicles')





from SUAVE.Core import Units
from SUAVE.Core import (
    Data, Container,
)
def main():

    from Boeing_747 import vehicle_setup, configs_setup
    vehicle = vehicle_setup()
    configs = configs_setup(vehicle)
    Mach                          = np.array([0.198])
    
    #conditions object used to create mission-like structure
    conditions                    = Data()
    conditions.weights            = Data()
    conditions.lift_curve_slope   = configs.base.wings['main_wing'].CL_alpha 
    conditions.weights.total_mass = np.array([[vehicle.mass_properties.max_takeoff]]) 
    conditions.aerodynamics       = Data()
    conditions.aerodynamics.angle_of_attack = 0.
   
    #print configuration
    cm_a           = taw_cmalpha(vehicle,Mach,conditions,configs.base)[0]
    expected       = -1.56222373 #Should be -1.45
    error          = Data()
    error.cm_a_747 = (cm_a - expected)/expected
    
    
    from Beech_99 import vehicle_setup, configs_setup
    vehicle = vehicle_setup()
    configs = configs_setup(vehicle)
    Mach    = np.array([0.152])
    
    #conditions object used to create mission-like structure
    conditions                    = Data()
    conditions.weights            = Data()
    conditions.lift_curve_slope   = configs.base.wings['main_wing'].CL_alpha 
    conditions.weights.total_mass = np.array([[vehicle.mass_properties.max_takeoff]]) 
    conditions.aerodynamics       = Data()
    conditions.aerodynamics.angle_of_attack = 0.    
   
    
    #Method Test   
    #print configuration
    cm_a = taw_cmalpha(vehicle,Mach,conditions,configs.base)[0]
    expected = -2.48843437 #Should be -2.08
    error.cm_a_beech_99 = (cm_a - expected)/expected   
    
    
    from SIAI_Marchetti_S211 import vehicle_setup, configs_setup
    
    vehicle = vehicle_setup()
    configs = configs_setup(vehicle)
    Mach                          = np.array([0.111])
    #conditions object used to create mission-like structure
    conditions                    = Data()
    conditions.weights            = Data()
    conditions.lift_curve_slope   = configs.base.wings['main_wing'].CL_alpha 
    conditions.weights.total_mass = np.array([[vehicle.mass_properties.max_takeoff]]) 
    conditions.aerodynamics       = Data()
    conditions.aerodynamics.angle_of_attack = 0.    
   


    cm_a = taw_cmalpha(vehicle,Mach,conditions,configs.base)[0]
   
    expected = -0.54071741 #Should be -0.6
    error.cm_a_SIAI = (cm_a - expected)/expected
    print(error)
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)
        
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
