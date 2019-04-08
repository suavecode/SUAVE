## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# wave_drag_volume.py
# 
# Created:  Jun 2014, T. MacDonald
# Modified: Feb 2019, T. MacDonald

import numpy as np
from SUAVE.Core import Units
from .Cubic_Spline_Blender import Cubic_Spline_Blender

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def wave_drag_volume(vehicle,mach,scaling_factor):
    """Computes the volume drag

    Assumptions:
    Basic fit

    Source:
    D. Raymer, Aircraft Design: A Conceptual Approach, Fifth Ed. pg. 448-449

    Inputs:
    vehicle.
      wings.main_wing.sweeps.leading_edge [rad]
      total_length                        [m]
      maximum_cross_sectional_area        [m^2]
      reference_area                      [m^2]
      
    Outputs:
    vehicle_wave_drag                     [Unitless]

    Properties Used:
    N/A
    """    
    LE_sweep = vehicle.wings.main_wing.sweeps.leading_edge / Units.deg
    L        = vehicle.total_length
    Ae       = vehicle.maximum_cross_sectional_area
    S        = vehicle.reference_area
    
    # Compute sears-hack D/q
    Dq_SH = 9*np.pi/2*(Ae/L)*(Ae/L)
    
    spline = Cubic_Spline_Blender(1.2,1.3)
    h00 = lambda M:spline.compute(M)    
    
    # Compute full vehicle D/q
    Dq_vehicle           = np.zeros_like(mach)
    Dq_vehicle_simpified = np.zeros_like(mach)
    
    Dq_vehicle[mach>=1.2] = scaling_factor*(1-0.2*(mach[mach>=1.2]-1.2)**0.57*(1-np.pi*LE_sweep**.77/100))*Dq_SH
    Dq_vehicle_simpified  = scaling_factor*Dq_SH
    
    Dq_vehicle = Dq_vehicle_simpified*h00(mach) + Dq_vehicle*(1-h00(mach))
    
    CD_c_vehicle = Dq_vehicle/S
    
    return CD_c_vehicle