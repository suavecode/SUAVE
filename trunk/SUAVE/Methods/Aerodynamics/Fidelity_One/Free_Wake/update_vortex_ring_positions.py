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
def update_vortex_ring_positions(prop, wVD_collapsed, VD, dt,quasi_wake=True): 
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
    
    Vinf  = prop.outputs.velocity
    
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
    V_avg = (V_A1+V_A2+V_B1+V_B2)/4

    Vx = Vinf[0,0] - V_avg[:,:,0]
    Vy = Vinf[0,1] - V_avg[:,:,1]
    Vz = Vinf[0,2] - V_avg[:,:,2]
    
    # Translate from velocity frame to rotor frame
    rot_to_body = prop.prop_vel_to_body()
    Vxp, Vyp, Vzp = rotate_frames(Vx, Vy, Vz, rot_to_body)
    
    if quasi_wake:
        # Update all panel points using averaged induced velocities
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
    else:
        # update point positions based on their local velocity, centers updated with the average velocities
        Vxa1 = Vinf[0,0] - V_A1[:,:,0]
        Vya1 = Vinf[0,1] - V_A1[:,:,1]
        Vza1 = Vinf[0,2] - V_A1[:,:,2]
        Vxa1, Vya1, Vza1 = rotate_frames(Vxa1, Vya1, Vza1, rot_to_body)
        
        Vxa2 = Vinf[0,0] - V_A2[:,:,0]
        Vya2 = Vinf[0,1] - V_A2[:,:,1]
        Vza2 = Vinf[0,2] - V_A2[:,:,2]
        Vxa2, Vya2, Vza2 = rotate_frames(Vxa2, Vya2, Vza2, rot_to_body)        
        
        Vxb1 = Vinf[0,0] - V_B1[:,:,0]
        Vyb1 = Vinf[0,1] - V_B1[:,:,1]
        Vzb1 = Vinf[0,2] - V_B1[:,:,2]
        Vxb1, Vyb1, Vzb1 = rotate_frames(Vxb1, Vyb1, Vzb1, rot_to_body)        
        
        Vxb2 = Vinf[0,0] - V_B2[:,:,0]
        Vyb2 = Vinf[0,1] - V_B2[:,:,1]
        Vzb2 = Vinf[0,2] - V_B2[:,:,2]
        Vxb2, Vyb2, Vzb2 = rotate_frames(Vxb2, Vyb2, Vzb2, rot_to_body)      
        
        wVD_collapsed.XA1 += Vxa1*dt
        wVD_collapsed.XA2 += Vxa2*dt
        wVD_collapsed.XB1 += Vxb1*dt
        wVD_collapsed.XB2 += Vxb2*dt
        wVD_collapsed.XC  += Vxp*dt
    
        
        wVD_collapsed.YA1 += Vya1*dt
        wVD_collapsed.YA2 += Vya2*dt
        wVD_collapsed.YB1 += Vyb1*dt
        wVD_collapsed.YB2 += Vyb2*dt
        wVD_collapsed.YC  += Vyp*dt
    
        
        wVD_collapsed.ZA1 += Vza1*dt
        wVD_collapsed.ZA2 += Vza2*dt
        wVD_collapsed.ZB1 += Vzb1*dt
        wVD_collapsed.ZB2 += Vzb2*dt
        wVD_collapsed.ZC  += Vzp*dt        
        

    return wVD_collapsed

def rotate_frames(Vx, Vy, Vz, rot_to_body):
    Vxp = Vx*rot_to_body[2,2] + Vz*rot_to_body[2,0]  # x in prop frame points downstream
    Vyp = Vy*rot_to_body[1,1]                        # y in prop frame along propeller plane
    Vzp = Vz*rot_to_body[0,0] + Vx*rot_to_body[0,2]  # z in prop frame along propeller plane
    
    return Vxp, Vyp, Vzp