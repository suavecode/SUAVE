## @ingroupMethods-Noise-Boom
# lift_equivalent_area.py
# 
# Created:  Jul 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------


import numpy as np
from SUAVE.Core import Data, Units

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift import VLM


# ----------------------------------------------------------------------
#   Equivalent Area from lift for Sonic Boom
# ----------------------------------------------------------------------
## @ingroupMethods-Noise-Boom
def lift_equivalent_area(config,analyses,conditions):
    
    S                                  = config.reference_area
    q                                  = conditions.freestream.dynamic_pressure
    mach                               = conditions.freestream.mach_number
    settings                           = Data()
    settings.number_spanwise_vortices  = analyses.aerodynamics.settings.number_spanwise_vortices
    settings.number_chordwise_vortices = analyses.aerodynamics.settings.number_chordwise_vortices
    settings.model_fuselage            = True
    settings.propeller_wake_model      = None

    CL, CDi, CM, CL_wing, CDi_wing, cl_y , cdi_y , CP ,Velocity_Profile = VLM(conditions, settings, config)
    
    VD = analyses.aerodynamics.geometry.vortex_distribution
    
    areas      = VD.panel_areas
    normal_vec = VD.unit_normals
    XC         = VD.XC
    ZC         = VD.ZC
    z_comp     = normal_vec[:,2]

    # The 2 is used because the CP acts on both the top and bottom of the panel
    lift_force_per_panel = 2*CP*q*z_comp*areas
    
    # Mach angle
    mach_angle = np.arcsin(1/mach[0])
    
    # Shift all points onto the X axis
    X_shift = XC + ZC*np.tan(mach_angle)
    
    # Order the values
    sort_order = np.argsort(X_shift)
    X  = np.take(X_shift,sort_order)
    Y  = np.take(lift_force_per_panel, sort_order)

    u, inv = np.unique(X, return_inverse=True)
    sums   = np.zeros(len(u), dtype=Y.dtype) 
    np.add.at(sums, inv, Y) 
    
    lift_forces = sums
    X_locations = u
    
    summed_lift_forces = np.cumsum(lift_forces)
    
    beta = np.sqrt(conditions.freestream.mach_number[0][0]**2 -1)
    
    Ae_lift_at_each_x = (beta/(2*q[0]))*summed_lift_forces
    
    X_max  = config.total_length
    
    X_locs = np.concatenate([[0],X_locations,[X_max]])
    AE_x   = np.concatenate([[0],Ae_lift_at_each_x,[Ae_lift_at_each_x[-1]]])
    
    return X_locs, AE_x