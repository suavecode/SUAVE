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
def compute_wake_induced_velocity(WD,VD,cpts,sigma=0.11):  
    """ This computes the velocity induced by the fixed helical wake
    on lifting surface control points

    Assumptions:  
    
    Source:   
    
    Inputs: 
    WD     - helical wake distribution points               [Unitless] 
    VD     - vortex distribution points on lifting surfaces [Unitless] 
    cpts   - control points in segment                      [Unitless] 

    Properties Used:
    N/A
    """    
    
    # control point, time step , blade number , location on blade 
    num_v_cpts = len(WD.XA1[0,:])  
    num_w_cpts = VD.n_cp
    ones       = np.ones((cpts,1,1))  
    
    dtype = np.float64
    
    WXA1  = np.repeat(np.atleast_3d(WD.XA1).astype(dtype), num_w_cpts , axis = 2)
    WYA1  = np.repeat(np.atleast_3d(WD.YA1).astype(dtype), num_w_cpts , axis = 2)     
    WZA1  = np.repeat(np.atleast_3d(WD.ZA1).astype(dtype), num_w_cpts , axis = 2)
    WXA2  = np.repeat(np.atleast_3d(WD.XA2).astype(dtype), num_w_cpts , axis = 2)
    WYA2  = np.repeat(np.atleast_3d(WD.YA2).astype(dtype), num_w_cpts , axis = 2)
    WZA2  = np.repeat(np.atleast_3d(WD.ZA2).astype(dtype), num_w_cpts , axis = 2)
                           
    WXB1  = np.repeat(np.atleast_3d(WD.XB1).astype(dtype), num_w_cpts , axis = 2)
    WYB1  = np.repeat(np.atleast_3d(WD.YB1).astype(dtype), num_w_cpts , axis = 2)
    WZB1  = np.repeat(np.atleast_3d(WD.ZB1).astype(dtype), num_w_cpts , axis = 2)
    WXB2  = np.repeat(np.atleast_3d(WD.XB2).astype(dtype), num_w_cpts , axis = 2)
    WYB2  = np.repeat(np.atleast_3d(WD.YB2).astype(dtype), num_w_cpts , axis = 2)
    WZB2  = np.repeat(np.atleast_3d(WD.ZB2).astype(dtype), num_w_cpts , axis = 2)
    GAMMA = np.repeat(np.atleast_3d(WD.GAMMA).astype(dtype), num_w_cpts , axis = 2)
    
    XC    = np.repeat(np.atleast_2d(VD.XC*ones).astype(dtype), num_v_cpts , axis = 1)
    YC    = np.repeat(np.atleast_2d(VD.YC*ones).astype(dtype), num_v_cpts , axis = 1)
    ZC    = np.repeat(np.atleast_2d(VD.ZC*ones).astype(dtype), num_v_cpts , axis = 1)
    
    # -------------------------------------------------------------------------------------------
    # Compute velocity induced by horseshoe vortex segments on every control point by every panel
    # -------------------------------------------------------------------------------------------     
    # Create empty data structure
    V_ind = np.zeros((cpts,VD.n_cp,3))
     
    #compute vortex strengths for every control point on wing 
    # this loop finds the strength of one ring only on entire control points on wing 
    # compute influence of bound vortices 
    _ , res_C_AB = vortex(XC, YC, ZC, WXA1, WYA1, WZA1, WXB1, WYB1, WZB1,sigma,GAMMA,bv=True,VD=VD) 
    C_AB         = np.transpose(res_C_AB,axes=[1,2,3,0]) 
    
    # compute influence of 3/4 right legs 
    _ , res_C_BC = vortex(XC, YC, ZC, WXB1, WYB1, WZB1, WXB2, WYB2, WZB2,sigma,GAMMA) 
    C_BC         = np.transpose(res_C_BC,axes=[1,2,3,0]) 
    
    # compute influence of bound vortices  
    _ , res_C_CD = vortex(XC, YC, ZC, WXB2, WYB2, WZB2, WXA2, WYA2, WZA2,sigma,GAMMA) 
    C_CD         = np.transpose(res_C_CD,axes=[1,2,3,0])
    
    # compute influence of 3/4 left legs  
    _ , res_C_DA = vortex(XC, YC, ZC, WXA2, WYA2, WZA2, WXA1, WYA1, WZA1,sigma,GAMMA) 
    C_DA         = np.transpose(res_C_DA,axes=[1,2,3,0]) 
    
    # Add all the influences together
    V_ind =  np.sum(C_AB +  C_BC  + C_CD + C_DA, axis = 1)
    
    return V_ind
  
  
# -------------------------------------------------------------------------------
# vortex strength computation
# -------------------------------------------------------------------------------
## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def vortex(X,Y,Z,X1,Y1,Z1,X2,Y2,Z2,sigma, GAMMA = 1, bv=False,VD=None,use_regularization_kernal=True):
    """ This computes the velocity induced on a control point by a segment
    of a horseshoe vortex from point 1 to point 2 
    Assumptions:  
    None 
    
    Source: 
    Low-Speed Aerodynamics, Second Edition by Joseph katz, Allen Plotkin
    Pgs. 584(Literature), 579-586 (Fortran Code implementation)
    
    Inputs:
    GAMMA       - propeller/rotor circulation
    [X,Y,Z]     - location of control point  
    [X1,Y1,Z1]  - location of point 1 
    [X2,Y2,Z2]  - location of point 2
    Properties Used:
    N/A
    
    """      

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
    SQUARE[SQUARE==0] = 1e-8
    R1     = np.sqrt(np.square(X_X1) + np.square(Y_Y1) + np.square(Z_Z1)) 
    R2     = np.sqrt(np.square(X_X2) + np.square(Y_Y2) + np.square(Z_Z2)) 
    R0R1   = X2_X1*X_X1 + Y2_Y1*Y_Y1 + Z2_Z1*Z_Z1
    R0R2   = X2_X1*X_X2 + Y2_Y1*Y_Y2 + Z2_Z1*Z_Z2
    RVEC   = np.array([R1R2X,R1R2Y,R1R2Z])
    COEF   = (1/(4*np.pi))*(RVEC/SQUARE) * (R0R1/R1 - R0R2/R2)    
    
    if use_regularization_kernal:
        COEF = regularization_kernel(COEF, sigma)
    
    if bv:
        # ignore the row of panels corresponding to the lifting line of the rotor
        COEF_new = np.reshape(COEF[0,:,:,0],np.shape(VD.Wake.XA1))
        m = np.shape(VD.Wake.XA1)[0]
        
        lifting_line_panels = np.zeros_like(COEF_new,dtype=bool)
        lifting_line_panels[:,:,:,:,0] = True
        lifting_line_panels_compressed = np.reshape(lifting_line_panels, (m,np.size(lifting_line_panels[0,:,:,:,:])))
        
        COEF[:,lifting_line_panels_compressed,:] = 0
        
    V_IND  = GAMMA * COEF
    
    return COEF , V_IND  

def regularization_kernel(COEF_in, sigma):
    """
    Inputs:
       COEF    Biot-Savart Kernel
       sigma   regularization radius
    Outputs:
       KAPPA   Regularization Kernel
    
    """
    COEF       = COEF_in.astype(np.float32)
    COEF_MAG   = np.abs(COEF)
    
    # Make sure the magnitude doesn't go to zero
    COEF_MAG[np.isclose(COEF_MAG,0,atol=1e-8)] = 1e-8

    R_square   = 1/(4*np.pi*COEF_MAG)
    R          = np.sqrt(R_square)
    R_sigma_sq = R_square/np.square(sigma)
    
    NUM = R*(R_sigma_sq + (5/2))
    DEN = 4*np.pi*(sigma**3)*np.power(R_sigma_sq + 1,5/2)
    
    DEN[DEN==0.] = 1e-8
    
    KAPPA = (NUM / DEN) * np.sign(COEF)
    
    return KAPPA