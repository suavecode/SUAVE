## @ingroupMethods-Noise-Boom
# lift_equivalent_area.py
# 
# Created:  Sep 2020, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift import VLM

# ----------------------------------------------------------------------
#   Equivalent Area from lift for Sonic Boom
# ----------------------------------------------------------------------
## @ingroupMethods-Noise-Boom
def lift_equivalent_area(config,analyses,conditions):
    """ This method calculates the lift equivalent area for a vehicle for sonic boom analysis.
    
        Assumptions:
        X_locs is the location where lift values are taken on the x-axis
        AE_x is the lift equivalent area
        CP is the coefficient of pressure at the flight conditions
        
        
        Source:
        N/A
        
        Inputs:
        config                                [vehicle]
        conditions
              .freestream.dynamic_pressure    [pascals]
              .freestream.mach_number         [-]
              .aerodynamics.angle_of_attack   [radians]
        analyses.aerodynamics.settings        [needed for discretization]
        config.total_length                   [m]

        Outputs:
        X_locs                                [m]
        AE_x                                  [m^2]
        CP                                    [pascals]
        
        Properties Used:
        N/A
    """       
    
    S            = config.reference_area
    q            = conditions.freestream.dynamic_pressure
    mach         = conditions.freestream.mach_number
    aoa          = conditions.aerodynamics.angle_of_attack
    settings     = analyses.aerodynamics.process.compute.lift.inviscid_wings.settings
    length       = config.total_length
    
    results = VLM(conditions, settings, config)
    CP = results.CP
    
    VD = analyses.aerodynamics.geometry.vortex_distribution
    
    areas      = VD.panel_areas
    normal_vec = VD.normals
    XC         = VD.XC
    ZC         = VD.ZC
    z_comp     = normal_vec[:,2]

    lift_force_per_panel = CP[0,:]*q*z_comp*areas*np.cos(aoa[0])
    
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
    
    X_max  = length*1.25
    
    X_locs = np.concatenate([[0],X_locations,[X_max]])
    AE_x   = np.concatenate([[0],Ae_lift_at_each_x,[Ae_lift_at_each_x[-1]]])
    
    return X_locs, AE_x, CP