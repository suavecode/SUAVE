## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# wave_drag_volume.py
# 
# Created:  Jun 2014, T. MacDonald
# Modified: Feb 2019, T. MacDonald
#           Jan 2020, T. MacDonald
#           Apr 2020, M. Clarke

import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Utilities.Cubic_Spline_Blender import Cubic_Spline_Blender
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep
from SUAVE.Components.Wings import Main_Wing

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def wave_drag_volume_sears_haack_3(vehicle,mach,scaling_factor):
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
    
    # Compute drag from sears-haack type III body
    # Source formula uses front projected area as a reference
    CD = 3/2*np.pi*np.pi*(d/L)*(d/L)*Amax/S
    # Scale to account for non-ideal shaping
    CD_scaled = CD*scaling_factor
    
    return CD_scaled