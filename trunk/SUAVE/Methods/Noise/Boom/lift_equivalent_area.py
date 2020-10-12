## @ingroupMethods-Noise-Boom
# lift_equivalent_area.py
# 
# Created:  Jul 2014, A. Wendorff
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np
from SUAVE.Core import Data, Units

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift import VLM


# ----------------------------------------------------------------------
#   Equivalent Area from lift for Sonic Boom
# ----------------------------------------------------------------------
## @ingroupMethods-Noise-Boom
def lift_equivalent_area(config,analyses):
    
    conditions = Data()
    settings   = Data()
    settings.number_spanwise_vortices  = analyses.aerodynamics.settings.number_spanwise_vortices
    settings.number_chordwise_vortices = analyses.aerodynamics.settings.number_chordwise_vortices
    settings.propeller_wake_model      = None
    conditions.aerodynamics = Data()
    conditions.freestream   = Data()
    conditions.aerodynamics.angle_of_attack = np.array([[0.03270489]])
    conditions.freestream.mach_number       = np.array([[2.02]])
    conditions.freestream.velocity          = np.array([[596.04038125]])
    q                                       = np.array([[24415.0099073]])
    
    CL, CDi, CM, CL_wing, CDi_wing, cl_y , cdi_y , CP ,Velocity_Profile = VLM(conditions, settings, config)
    
    S   =  config.reference_area
    
    VD = analyses.aerodynamics.geometry.vortex_distribution
    
    # delete MCM from VD data structure since it consumes memory
    delattr(VD, 'MCM')       
    
    areas      = VD.panel_areas
    normal_vec = VD.unit_normals
    XC         = VD.XC
    z_comp     = normal_vec[:,2]

    # Why do I need a 2 here????? But it works perfectly otherwise!
    lift_force_per_panel = 2*CP*q*z_comp*areas
    
    L_CP = np.sum(lift_force_per_panel)
    
    # Check the lift forces
    L_CL = q*CL*S
    
    # Order the values
    sort_order = np.argsort(XC)
    X  = np.take(XC,sort_order)
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

    
    fig  = plt.figure('boom')
    axes = fig.add_subplot(1,1,1)
    axes.plot(X_locs,AE_x)
    
    
    return X_locs, AE_x