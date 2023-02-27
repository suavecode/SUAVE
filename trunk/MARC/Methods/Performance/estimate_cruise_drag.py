## @ingroup Methods-Performance
# estimate_cruise_drag.py
#
# Created: Feb 2023, M. Clarke

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
import MARC
import numpy as np 

#------------------------------------------------------------------------------
# Cruise Drag Function
#------------------------------------------------------------------------------

## @ingroup Methods-Performance
def estimate_cruise_drag(vehicle,altitude,speed,lift_coefficient = 0.5 ,profile_drag = 0.05):

    """Calculates the drag force of an aircraft at a given altitude and  cruising speed.

        Sources:
        N/A

        Assumptions:
        None 

        Inputs:
            vehicle                         vehicle    
            altitude                        cruise altitude          [m] 
            speed                           cruise speed             [m/s] 
            lift_coefficient                cruise lift coefficient  [-] 
            
        Outputs: 
            Drag                            cruise drag              [N]
    """ 
       
    atmo = MARC.Analyses.Atmospheric.US_Standard_1976()
    rho  = atmo.compute_values(altitude,0.).density[0][0]  
    S    = vehicle.reference_area                     # reference area   
    AR   = vehicle.wings.main_wing.aspect_ratio       # aspect ratio  
    Cdi  = lift_coefficient**2/(np.pi*AR*0.98)        # induced drag
    Cd   = profile_drag + Cdi                         # total drag 
    Drag = S * (0.5*rho*speed**2 )*Cd                 # cruise drag 
    
    return Drag