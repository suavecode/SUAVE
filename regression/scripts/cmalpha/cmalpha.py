# test_cmalpha.py
# Tim Momose, April 2014
# Reference: Aircraft Dynamics: from Modeling to Simulation, by M. R. Napolitano

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_mac import trapezoid_mac
#from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_ac_x import trapezoid_ac_x
#from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approsimations.Supporting_Functions.extend_to_ref_area import extend_to_ref_area
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cmalpha import taw_cmalpha
from SUAVE.Methods.Geometry.Three_Dimensional.compute_span_location_from_chord_length import compute_span_location_from_chord_length
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
   
    #print configuration
    cm_a           = taw_cmalpha(vehicle,Mach,conditions,configs.base)
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
   
    
    #Method Test   
    #print configuration
    cm_a = taw_cmalpha(vehicle,Mach,conditions,configs.base)
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
   
    
    
    cm_a = taw_cmalpha(vehicle,Mach,conditions,configs.base)
   
    expected = -0.54071741 #Should be -0.6
    error.cm_a_SIAI = (cm_a - expected)/expected
    print error
    for k,v in error.items():
        assert(np.abs(v)<0.01)
        
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
