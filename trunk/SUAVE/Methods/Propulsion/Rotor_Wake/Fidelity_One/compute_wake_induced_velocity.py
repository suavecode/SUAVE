## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# compute_wake_induced_velocity.py
# 
# Created:  Sep 2020, M. Clarke 
# Modified: Dec 2021, R. Erhard
#           May 2022, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import jax.numpy as jnp

# ----------------------------------------------------------------------
#  Compute Wake Induced Velocity
# ----------------------------------------------------------------------
## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def compute_wake_induced_velocity(WD,VD,cpts,azi_start_idx=0,sigma=0.11,suppress_root=False):  
    """ This computes the velocity induced by the Fidelity One semi-prescribed vortex wake (PVW)
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
    num_vortex_pts = len(WD.XA1[0,0,:])    # number of vortex points
    num_eval_pts   = VD.n_cp               # number of evaluation points
    
    dtype = jnp.float64

    # expand vortex points
    WXA1  = jnp.tile(WD.XA1.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WYA1  = jnp.tile(WD.YA1.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))     
    WZA1  = jnp.tile(WD.ZA1.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WXA2  = jnp.tile(WD.XA2.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WYA2  = jnp.tile(WD.YA2.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WZA2  = jnp.tile(WD.ZA2.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
                
    WXB1  = jnp.tile(WD.XB1.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WYB1  = jnp.tile(WD.YB1.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WZB1  = jnp.tile(WD.ZB1.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WXB2  = jnp.tile(WD.XB2.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WYB2  = jnp.tile(WD.YB2.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    WZB2  = jnp.tile(WD.ZB2.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    GAMMA = jnp.tile(WD.GAMMA.astype(dtype)[azi_start_idx,:,:,None], (1,1,num_eval_pts))
    
    # expand evaluation points
    XC    = jnp.tile(VD.XC.astype(dtype)[None,None,:],(cpts,num_vortex_pts,1))
    YC    = jnp.tile(VD.YC.astype(dtype)[None,None,:],(cpts,num_vortex_pts,1))
    ZC    = jnp.tile(VD.ZC.astype(dtype)[None,None,:],(cpts,num_vortex_pts,1))
    
    # -------------------------------------------------------------------------------------------
    # Compute velocity induced by horseshoe vortex segments on every control point by every panel
    # -------------------------------------------------------------------------------------------     
    # Create empty data structure
    V_ind = jnp.zeros((cpts,VD.n_cp,3))
     
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
    
    SQUARE = jnp.square(R1R2X) + jnp.square(R1R2Y) + jnp.square(R1R2Z)
    SQUARE = jnp.where(SQUARE==0,1e-8,SQUARE)
    R1     = jnp.sqrt(jnp.square(X_X1) + jnp.square(Y_Y1) + jnp.square(Z_Z1)) 
    R2     = jnp.sqrt(jnp.square(X_X2) + jnp.square(Y_Y2) + jnp.square(Z_Z2)) 
    R0R1   = X2_X1*X_X1 + Y2_Y1*Y_Y1 + Z2_Z1*Z_Z1
    R0R2   = X2_X1*X_X2 + Y2_Y1*Y_Y2 + Z2_Z1*Z_Z2
    RVEC   = jnp.array([R1R2X,R1R2Y,R1R2Z])
    COEF   = (1/(4*jnp.pi))*(RVEC/SQUARE) * (R0R1/R1 - R0R2/R2)    

    
    if use_regularization_kernal:
        COEF = regularization_kernel(COEF, sigma)

    if bv:
        # ignore the row of panels corresponding to the lifting line of the rotor
        COEF_new = jnp.reshape(COEF[0,:,:,0],jnp.shape(WD.reshaped_wake.XA1[0,:,:,:,:]))
        m        = jnp.shape(WD.reshaped_wake.XA1)[1]
        
        lifting_line_panels            = jnp.zeros_like(COEF_new,dtype=bool)
        lifting_line_panels            = lifting_line_panels.at[:,:,:,0].set(True)
        lifting_line_panels_compressed = jnp.reshape(lifting_line_panels, (m,jnp.size(lifting_line_panels[0,:,:,:])))
        lifting_line_panels_full       = lifting_line_panels_compressed[jnp.newaxis, ...,jnp.newaxis]
        lifting_line_panels_full       = jnp.broadcast_to(lifting_line_panels_full,COEF.shape)
        
        COEF = jnp.where(lifting_line_panels_full,0.,COEF)
    

    V_IND  = GAMMA * COEF
    
    return COEF , V_IND  

def row_reduction_summation(A):
    # sum along last axis
    sum_res = jnp.dot(A,jnp.ones(A.shape[-1])) # sum along axis
    
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
    COEF       = COEF_in.astype(jnp.float32)
    COEF_MAG   = jnp.abs(COEF)
    
    # Make sure the magnitude doesn't go to zero
    COEF_MAG = jnp.where(jnp.isclose(COEF_MAG,0,atol=1e-8),1e-8,COEF_MAG)

    R_square   = 1/(4*jnp.pi*COEF_MAG)
    R          = jnp.sqrt(R_square)
    R_sigma_sq = R_square/jnp.square(sigma)
    
    NUM = R*(R_sigma_sq + (5/2))
    DEN = 4*jnp.pi*(sigma**3)*jnp.power(R_sigma_sq + 1,5/2)
    
    DEN = jnp.where(DEN==0.,1e-8,DEN)
    
    KAPPA = (NUM / DEN) * jnp.sign(COEF)
    
    return KAPPA