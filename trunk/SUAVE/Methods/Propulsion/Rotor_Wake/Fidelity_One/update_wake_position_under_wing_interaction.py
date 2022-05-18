## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# update_wake_position_under_component_interaction.py
# 
# Created:  May 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np 

## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def update_wake_position_under_wing_interaction(wake, wing, VD_wing):  
    """
    Takes the vortex distribution of another aircraft component and evolves the rotor
    wake's vortex filament positions to account for the presence of the nearby component interaction.
    
    This is an approximation method intended to account for first order effects of interactions that
    will change the rotor wake shape. 
    
    Assumptions:
    N/A
    
    Source:
    N/A
    
    Inputs:
       wake           - a fidelity one rotor wake
       VD_component   - Vortex distribution associated with the component of interest
    
    Outputs:
       VD_wake        - Updated vortex distribution of the rotor wake due to component interaction
        
    """    

    # -------------------------------------------------------------------------------------------    
    #        Loop over each time step to update the rotor wake shape
    # -------------------------------------------------------------------------------------------    
    nts = wake.wake_settings.number_steps_per_rotation * wake.wake_settings.number_rotor_rotations
    for t in range(nts):
        # ------------------------------------------------------------------------------------------- 
        # Compute the velocity induced by VD_component at the trailing edge panels at time step t
        # ------------------------------------------------------------------------------------------- 
        panels = wake.vortex_distribution
        
        # Update the collocation points to compute the wing induced velocities at the wake panel TE
        VD_wing.XC = panels.XC
        
        # Compute the wing induced velocities at these control points
        compute_wing_induced_velocity(VD,mach,compute_EW=False)

        # -------------------------------------------------------------------------------------------         
        # Update the position of all panels in rotor wake
        # -------------------------------------------------------------------------------------------         
        # Translate the center of this row based on induced velocities
        
        # Rotate trailing edge points of this row's panels accordingly
        
        # Translate all downstream panels to connect with this row's trailing edge
    
    
    
    
    
    # control point, time step , blade number , location on blade 
    num_vortex_pts = len(WD.XA1[0,0,:])    # number of vortex points
    num_eval_pts   = VD.n_cp               # number of evaluation points
    
    dtype = np.float64

    # expand vortex points
    WXA1  = np.tile(WD.XA1.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WYA1  = np.tile(WD.YA1.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))     
    WZA1  = np.tile(WD.ZA1.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WXA2  = np.tile(WD.XA2.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WYA2  = np.tile(WD.YA2.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WZA2  = np.tile(WD.ZA2.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
                
    WXB1  = np.tile(WD.XB1.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WYB1  = np.tile(WD.YB1.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WZB1  = np.tile(WD.ZB1.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WXB2  = np.tile(WD.XB2.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WYB2  = np.tile(WD.YB2.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WZB2  = np.tile(WD.ZB2.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    GAMMA = np.tile(WD.GAMMA.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    
    # expand evaluation points
    XC    = np.tile(VD.XC.astype(dtype)[None,None,:],(cpts,num_vortex_pts,1))
    YC    = np.tile(VD.YC.astype(dtype)[None,None,:],(cpts,num_vortex_pts,1))
    ZC    = np.tile(VD.ZC.astype(dtype)[None,None,:],(cpts,num_vortex_pts,1))
    
    # -------------------------------------------------------------------------------------------
    # Compute velocity induced by horseshoe vortex segments on every control point by every panel
    # -------------------------------------------------------------------------------------------     
    # Create empty data structure
    V_ind = np.zeros((cpts,VD.n_cp,3))
     
    # compute influence of bound vortices 
    _ , res_C_AB = vortex(XC, YC, ZC, WXA1, WYA1, WZA1, WXB1, WYB1, WZB1,sigma,GAMMA,bv=True,WD=WD) 
    C_AB         = res_C_AB.transpose(1,3,0,2) 
    
    # compute influence of right vortex segment
    _ , res_C_BC = vortex(XC, YC, ZC, WXB1, WYB1, WZB1, WXB2, WYB2, WZB2,sigma,GAMMA)
    C_BC         = res_C_BC.transpose(1,3,0,2) 
    
    # compute influence of bottom vortex segment
    _ , res_C_CD = vortex(XC, YC, ZC, WXB2, WYB2, WZB2, WXA2, WYA2, WZA2,sigma,GAMMA) 
    C_CD         = res_C_CD.transpose(1,3,0,2) 
    
    # compute influence of left vortex segment 
    _ , res_C_DA = vortex(XC, YC, ZC, WXA2, WYA2, WZA2, WXA1, WYA1, WZA1,sigma,GAMMA) 
    C_DA         = res_C_DA.transpose(1,3,0,2) 
    
    # Add all the influences together
    V_ind =  row_reduction_summation(C_AB) + row_reduction_summation(C_BC)  + row_reduction_summation(C_CD) + row_reduction_summation(C_DA)   
    

    return V_ind
  
  
# -------------------------------------------------------------------------------
# vortex strength computation
# -------------------------------------------------------------------------------
## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def vortex(X,Y,Z,X1,Y1,Z1,X2,Y2,Z2,sigma, GAMMA = 1, bv=False,WD=None,use_regularization_kernal=True):
    """ This computes the velocity induced on a control point by a segment
    of a horseshoe vortex that points from point 1 to point 2 for a filament with
    positive vortex strength.
    
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
        COEF_new = np.reshape(COEF[0,:,:,0],np.shape(WD.reshaped_wake.XA1[0,:,:,:,:]))
        m = np.shape(WD.reshaped_wake.XA1)[1]
        
        lifting_line_panels = np.zeros_like(COEF_new,dtype=bool)
        lifting_line_panels[:,:,:,0] = True
        lifting_line_panels_compressed = np.reshape(lifting_line_panels, (m,np.size(lifting_line_panels[0,:,:,:])))
        
        COEF[:,lifting_line_panels_compressed,:] = 0
    

    V_IND  = GAMMA * COEF
    
    return COEF , V_IND  

def row_reduction_summation(A):
    # sum along last axis
    sum_res = A.dot(np.ones(A.shape[-1])) # sum along axis
    
    return sum_res

## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def regularization_kernel(COEF_in, sigma):
    """
    Regularization kernel used to prevent singularities
    
    Assumptions
    Spreads the vortex core over the radius sigma
    
    Source
       Winckelmans, "Topics in Vortex Methods for the Computation of Three-and Two-dimensional 
       Incompressible Unsteady Flows", 1989.
    
    Inputs:
       COEF    Biot-Savart Kernel
       sigma   regularization radius
    
    Outputs:
       KAPPA   Regularization Kernel
       
    Properties Used:
    N/A
    
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