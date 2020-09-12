## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_wake_induced_velocity.py
# 
# Created:  Sep 2020, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np 

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_wake_induced_velocity(WD,VD,cpts,ts,B,N):  
    
    # control point, time step , blade number , location on blade 
    num_v_cpts = len(WD.XA1[0,:])  
    num_w_cpts = VD.n_cp
    ones  = np.ones((cpts,1,1))    
    
    WXA1   =   np.repeat(np.atleast_3d(WD.XA1), num_w_cpts , axis = 2)    
    WYA1   =   np.repeat(np.atleast_3d(WD.YA1), num_w_cpts , axis = 2)     
    WZA1   =   np.repeat(np.atleast_3d(WD.ZA1), num_w_cpts , axis = 2)     
    WXA2   =   np.repeat(np.atleast_3d(WD.XA2), num_w_cpts , axis = 2)     
    WYA2   =   np.repeat(np.atleast_3d(WD.YA2), num_w_cpts , axis = 2)     
    WZA2   =   np.repeat(np.atleast_3d(WD.ZA2), num_w_cpts , axis = 2)     
                              
    WXB1   =   np.repeat(np.atleast_3d(WD.XB1), num_w_cpts , axis = 2)     
    WYB1   =   np.repeat(np.atleast_3d(WD.YB1), num_w_cpts , axis = 2)     
    WZB1   =   np.repeat(np.atleast_3d(WD.ZB1), num_w_cpts , axis = 2)     
    WXB2   =   np.repeat(np.atleast_3d(WD.XB2), num_w_cpts , axis = 2)     
    WYB2   =   np.repeat(np.atleast_3d(WD.YB2), num_w_cpts , axis = 2)     
    WZB2   =   np.repeat(np.atleast_3d(WD.ZB2), num_w_cpts , axis = 2)       
    GAMMA  =  np.repeat(np.atleast_3d(WD.GAMMA), num_w_cpts , axis = 2)   
    
    XC    =  np.repeat(np.atleast_2d(VD.XC*ones), num_v_cpts , axis = 1)   
    YC    =  np.repeat(np.atleast_2d(VD.YC*ones), num_v_cpts , axis = 1) 
    ZC    =  np.repeat(np.atleast_2d(VD.ZC*ones), num_v_cpts , axis = 1)
    
    # -------------------------------------------------------------------------------------------
    # Compute velocity induced by horseshoe vortex segments on every control point by every panel
    # -------------------------------------------------------------------------------------------     
    # Create empty data structure
    V_ind = np.zeros((cpts,VD.n_cp,3))
     
    #compute vortex strengths for every control point on wing 
    # this loop finds the strength of one ring only on entire control points on wing 
    # compute influence of bound vortices 
    C_AB = np.transpose(vortex(XC, YC, ZC, WXA1, WYA1, WZA1, WXB1, WYB1, WZB1,GAMMA),axes=[1,2,3,0]) 
    
    # compute influence of 3/4 left legs 
    C_BC = np.transpose(vortex(XC, YC, ZC, WXB1, WYB1, WZB1, WXB2, WYB2, WZB2,GAMMA),axes=[1,2,3,0]) 
    
    # compute influence of whole panel left legs  
    C_CD = np.transpose(vortex(XC, YC, ZC, WXB2, WYB2, WZB2, WXA2, WYA2, WZA2,GAMMA),axes=[1,2,3,0])
    
    # compute influence of 3/4 right legs  
    C_DA = np.transpose(vortex(XC, YC, ZC, WXA2, WYA2, WZA2, WXA1, WYA1, WZA1,GAMMA),axes=[1,2,3,0]) 
    
    # Add all the influences together
    V_ind =  np.sum(C_AB +  C_BC  + C_CD + C_DA, axis = 1)
    
    return V_ind

# -------------------------------------------------------------------------------
# vortex strength computation
# -------------------------------------------------------------------------------
def vortex(X,Y,Z,X1,Y1,Z1,X2,Y2,Z2,GAMMA):
    # reference: page 584 Low speed aerodynamics 
    # Take all the differences
    X_X1  = X-X1
    X_X2  = X-X2
    X2_X1 = X2-X1

    Y_Y1  = Y-Y1
    Y_Y2  = Y-Y2
    Y2_Y1 = Y2-Y1

    Z_Z1  = Z-Z1
    Z_Z2  = Z-Z2 
    Z2_Z1 = Z2-Z1 

    R1R2X  = Y_Y1*Z_Z2 - Z_Z1*Y_Y2 
    R1R2Y  = Z_Z1*X_X2 - X_X1*Z_Z2
    R1R2Z  = X_X1*Y_Y2 - Y_Y1*X_X2
    SQUARE = np.square(R1R2X) + np.square(R1R2Y) + np.square(R1R2Z)
    SQUARE[SQUARE==0] = 1e-32
    R1    = np.sqrt(np.square(X_X1) + np.square(Y_Y1) + np.square(Z_Z1)) 
    R2    = np.sqrt(np.square(X_X2) + np.square(Y_Y2) + np.square(Z_Z2)) 
    R0R1  = X2_X1*X_X1 + Y2_Y1*Y_Y1 + Z2_Z1*Z_Z1
    R0R2  = X2_X1*X_X2 + Y2_Y1*Y_Y2 + Z2_Z1*Z_Z2
    RVEC  = np.array([R1R2X,R1R2Y,R1R2Z])
    COEF  = GAMMA /(4*np.pi*SQUARE) * (R0R1/R1 - R0R2/R2)    
    V_IND = RVEC * COEF 
    
    return V_IND 