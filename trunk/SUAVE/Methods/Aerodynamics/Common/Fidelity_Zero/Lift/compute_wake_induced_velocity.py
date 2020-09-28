## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_wake_induced_velocity.py
# 
# Created:  Sep 2020, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wing_induced_velocity import vortex 

# package imports
import numpy as np 

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift 
def compute_wake_induced_velocity(WD,VD,cpts):  
    """ This computes the velocity induced by the fixed helical wake
    on lifting surface control points

    Assumptions:  
    
    Source:   
    
    Inputs: 
    WD     - helical wake distribution points               [Unitless] 
    VD     - vortex distribution points on lifting surfaces [Unitless] 
    cpts   - control points in segemnt                     [Unitless] 

    Properties Used:
    N/A
    """    
    
    # control point, time step , blade number , location on blade 
    num_v_cpts = len(WD.XA1[0,:])  
    num_w_cpts = VD.n_cp
    ones       = np.ones((cpts,1,1))    
    
    WXA1  = np.repeat(np.atleast_3d(WD.XA1), num_w_cpts , axis = 2)    
    WYA1  = np.repeat(np.atleast_3d(WD.YA1), num_w_cpts , axis = 2)     
    WZA1  = np.repeat(np.atleast_3d(WD.ZA1), num_w_cpts , axis = 2)     
    WXA2  = np.repeat(np.atleast_3d(WD.XA2), num_w_cpts , axis = 2)     
    WYA2  = np.repeat(np.atleast_3d(WD.YA2), num_w_cpts , axis = 2)     
    WZA2  = np.repeat(np.atleast_3d(WD.ZA2), num_w_cpts , axis = 2)     
                           
    WXB1  = np.repeat(np.atleast_3d(WD.XB1), num_w_cpts , axis = 2)     
    WYB1  = np.repeat(np.atleast_3d(WD.YB1), num_w_cpts , axis = 2)     
    WZB1  = np.repeat(np.atleast_3d(WD.ZB1), num_w_cpts , axis = 2)     
    WXB2  = np.repeat(np.atleast_3d(WD.XB2), num_w_cpts , axis = 2)     
    WYB2  = np.repeat(np.atleast_3d(WD.YB2), num_w_cpts , axis = 2)     
    WZB2  = np.repeat(np.atleast_3d(WD.ZB2), num_w_cpts , axis = 2)       
    GAMMA = np.repeat(np.atleast_3d(WD.GAMMA), num_w_cpts , axis = 2)   
    
    XC    = np.repeat(np.atleast_2d(VD.XC*ones), num_v_cpts , axis = 1)   
    YC    = np.repeat(np.atleast_2d(VD.YC*ones), num_v_cpts , axis = 1) 
    ZC    = np.repeat(np.atleast_2d(VD.ZC*ones), num_v_cpts , axis = 1)
    
    # -------------------------------------------------------------------------------------------
    # Compute velocity induced by horseshoe vortex segments on every control point by every panel
    # -------------------------------------------------------------------------------------------     
    # Create empty data structure
    V_ind = np.zeros((cpts,VD.n_cp,3))
     
    #compute vortex strengths for every control point on wing 
    # this loop finds the strength of one ring only on entire control points on wing 
    # compute influence of bound vortices 
    _ , res_C_AB = vortex(XC, YC, ZC, WXA1, WYA1, WZA1, WXB1, WYB1, WZB1,GAMMA) 
    C_AB         = np.transpose(res_C_AB,axes=[1,2,3,0]) 
    
    # compute influence of 3/4 left legs 
    _ , res_C_BC = vortex(XC, YC, ZC, WXB1, WYB1, WZB1, WXB2, WYB2, WZB2,GAMMA) 
    C_BC         = np.transpose(res_C_BC,axes=[1,2,3,0]) 
    
    # compute influence of whole panel left legs  
    _ , res_C_CD = vortex(XC, YC, ZC, WXB2, WYB2, WZB2, WXA2, WYA2, WZA2,GAMMA) 
    C_CD         = np.transpose(res_C_CD,axes=[1,2,3,0])
    
    # compute influence of 3/4 right legs  
    _ , res_C_DA = vortex(XC, YC, ZC, WXA2, WYA2, WZA2, WXA1, WYA1, WZA1,GAMMA) 
    C_DA         = np.transpose(res_C_DA,axes=[1,2,3,0]) 
    
    # Add all the influences together
    V_ind =  np.sum(C_AB +  C_BC  + C_CD + C_DA, axis = 1)
    
    return V_ind
  