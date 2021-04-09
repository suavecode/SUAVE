## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# wave_drag_volume_raymer.py
# 
# Created:  Jun 2014, T. MacDonald
# Modified: Feb 2019, T. MacDonald
#           Jan 2020, T. MacDonald
#           Apr 2020, M. Clarke
#           Feb 2021, T. MacDonald

import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Utilities.Cubic_Spline_Blender import Cubic_Spline_Blender
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep
from SUAVE.Components.Wings import Main_Wing

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def wave_drag_volume_raymer(vehicle,mach,scaling_factor):
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
    
    num_main_wings = 0
    for wing in vehicle.wings:
        if isinstance(wing,Main_Wing):
            main_wing = wing
            num_main_wings += 1
        if num_main_wings > 1:
            raise NotImplementedError('This function is not designed to handle multiple main wings.')
    
    main_wing = vehicle.wings.main_wing
    # estimation of leading edge sweep if not defined 
    if main_wing.sweeps.leading_edge == None:                           
        main_wing.sweeps.leading_edge  = convert_sweep(main_wing,old_ref_chord_fraction = 0.25 ,new_ref_chord_fraction = 0.0) 
        
    LE_sweep = main_wing.sweeps.leading_edge / Units.deg
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