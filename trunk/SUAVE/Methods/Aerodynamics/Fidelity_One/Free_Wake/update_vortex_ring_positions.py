## @ingroup Methods-Aerodynamics-Common-Fidelity_One-Free_Wake
#  update_vortex_ring_positions.py
# 
# Created:  Oct 2021, R. ERhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from SUAVE.Core import Data 
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wake_contraction_matrix import compute_wake_contraction_matrix
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wake_induced_velocity import compute_wake_induced_velocity
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry   

from copy import deepcopy

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift   
def update_vortex_ring_positions(prop, wVD_collapsed, VD, dt): 
    """
    Computes influence of all ring vortices on each other, updates positions accordinly
    
    Inputs:
       wVD         - wake vortex distribution
       dt          - time step
       
    Outputs:
       wVD         - new total wake vortex distribution with updated positions
    
    """
    #VD.XC = deepcopy(wVD_collapsed.XC)
    #VD.YC = deepcopy(wVD_collapsed.YC)
    #VD.ZC = deepcopy(wVD_collapsed.ZC)
    
    #-----------------------------------------------
    # Evaluate at all A1 points on panels
    #-----------------------------------------------
    VD.XC = deepcopy(wVD_collapsed.XA1)
    VD.YC = deepcopy(wVD_collapsed.YA1)
    VD.ZC = deepcopy(wVD_collapsed.ZA1)    
    VD.n_cp = wVD_collapsed.n_cp
    
    # compute vortex-induced velocities at centers of each vortex ring
    V_A1 = compute_wake_induced_velocity(wVD_collapsed, VD, cpts=1)
    

    #-----------------------------------------------
    # Evaluate at all A2 points on panels
    #-----------------------------------------------
    VD.XC = deepcopy(wVD_collapsed.XA2)
    VD.YC = deepcopy(wVD_collapsed.YA2)
    VD.ZC = deepcopy(wVD_collapsed.ZA2)    
    VD.n_cp = wVD_collapsed.n_cp
    
    # compute vortex-induced velocities at centers of each vortex ring
    V_A2 = compute_wake_induced_velocity(wVD_collapsed, VD, cpts=1)    
    
    #-----------------------------------------------
    # Evaluate at all B1 points on panels
    #-----------------------------------------------
    VD.XC = deepcopy(wVD_collapsed.XB1)
    VD.YC = deepcopy(wVD_collapsed.YB1)
    VD.ZC = deepcopy(wVD_collapsed.ZB1)    
    VD.n_cp = wVD_collapsed.n_cp
    
    # compute vortex-induced velocities at centers of each vortex ring
    V_B1 = compute_wake_induced_velocity(wVD_collapsed, VD, cpts=1)    
    
    #-----------------------------------------------
    # Evaluate at all B2 points on panels
    #-----------------------------------------------
    VD.XC = deepcopy(wVD_collapsed.XB2)
    VD.YC = deepcopy(wVD_collapsed.YB2)
    VD.ZC = deepcopy(wVD_collapsed.ZB2)    
    VD.n_cp = wVD_collapsed.n_cp
    
    # compute vortex-induced velocities at centers of each vortex ring
    V_B2 = compute_wake_induced_velocity(wVD_collapsed, VD, cpts=1)        
    

    #-----------------------------------------------
    # Average
    #-----------------------------------------------   
    V_ind = (V_A1+V_A2+V_B1+V_B2)/4
    
    Vinf  = prop.outputs.velocity
    
    Vx = Vinf[0,0] - V_ind[:,:,0]
    Vy = Vinf[0,1] - V_ind[:,:,1]
    Vz = Vinf[0,2] - V_ind[:,:,2]
    
    # Translate from velocity frame to rotor frame
    rot_to_body = prop.prop_vel_to_body()
    
    Vxp = Vx*rot_to_body[2,2] + Vz*rot_to_body[2,0]  # x in prop frame points downstream
    Vyp = Vy*rot_to_body[1,1]                        # y in prop frame along propeller plane
    Vzp = Vz*rot_to_body[0,0] + Vx*rot_to_body[0,2]  # z in prop frame along propeller plane
    
    # Translate vortex rings with velocity Vx in x-direction
    
    wVD_collapsed.XA1 += Vxp*dt
    wVD_collapsed.XA2 += Vxp*dt
    wVD_collapsed.XB1 += Vxp*dt
    wVD_collapsed.XB2 += Vxp*dt
    wVD_collapsed.XC  += Vxp*dt

    
    wVD_collapsed.YA1 += Vyp*dt
    wVD_collapsed.YA2 += Vyp*dt
    wVD_collapsed.YB1 += Vyp*dt
    wVD_collapsed.YB2 += Vyp*dt
    wVD_collapsed.YC  += Vyp*dt

    
    wVD_collapsed.ZA1 += Vzp*dt
    wVD_collapsed.ZA2 += Vzp*dt
    wVD_collapsed.ZB1 += Vzp*dt
    wVD_collapsed.ZB2 += Vzp*dt
    wVD_collapsed.ZC  += Vzp*dt
    
    
    
    
    return wVD_collapsed