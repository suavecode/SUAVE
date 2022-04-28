## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# wave_drag_volume_sears_haack.py
# 
# Created:  Feb 2021, T. MacDonald
# Modified: 

import numpy as np

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def wave_drag_volume_sears_haack(vehicle, mach, scaling_factor, sears_haack_type = 3):
    """Computes the volume drag
    
    Assumptions:
    Basic fit
    
    Source:
    Sieron, Thomas R., et al. Procedures and design data for the formulation of aircraft 
    configurations. WRIGHT LAB WRIGHT-PATTERSON AFB OH, 1993. Page B-3
    
    Inputs:
    vehicle.
      total_length                        [m]
      maximum_cross_sectional_area        [m^2]
      reference_area                      [m^2]
      
    Outputs:
    vehicle_wave_drag                     [Unitless]
    
    Properties Used:
    N/A
    """

    L        = vehicle.total_length
    Amax     = vehicle.maximum_cross_sectional_area
    S        = vehicle.reference_area

    rmax     = np.sqrt(Amax/np.pi)
    d        = rmax*2

    if sears_haack_type == 3:
        # Compute drag from Sears-Haack type III body
        # Source formula uses front projected area as a reference
        CD = 3/2*np.pi*np.pi*(d/L)*(d/L)*Amax/S
    else:
        raise NotImplementedError # can add other Sears-Haack types here
        
    # Scale to account for non-ideal shaping
    CD_scaled = CD*scaling_factor

    return CD_scaled 