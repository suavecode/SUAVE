## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_induced_velocity_matrix.py
# 
# Created:  May 2018, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import SUAVE
import numpy as np
from SUAVE.Core import Units , Data

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_induced_velocity_matrix(data,n_sw,n_cw,theta_w,mach):

    # unpack 
    ctrl_pts = len(theta_w)
    ones    = np.atleast_3d(np.ones_like(theta_w))
 
    # Prandtl Glauret Transformation for subsonic
    inv_root_beta = np.zeros_like(mach)
    inv_root_beta[mach<1] = 1/np.sqrt(1-mach[mach<1]**2)     
    inv_root_beta[mach>1] = 1/np.sqrt(mach[mach>1]**2-1) 
    if np.any(mach==1):
        raise('Mach of 1 cannot be used in building compressibiliy corrections.')
    inv_root_beta = np.atleast_3d(inv_root_beta)
     
    XAH   = np.atleast_3d(data.XAH*inv_root_beta)
    XAHbv = np.atleast_3d(data.XAH*inv_root_beta)
    YAH   = np.atleast_3d(data.YAH*ones)
    YAHbv = np.atleast_3d(data.YAH*ones)
    ZAH   = np.atleast_3d(data.ZAH*ones)
    ZAHbv = np.atleast_3d(data.ZAH*ones)
    XBH   = np.atleast_3d(data.XBH*inv_root_beta)
    XBHbv = np.atleast_3d(data.XBH*inv_root_beta)
    YBH   = np.atleast_3d(data.YBH*ones)
    YBHbv = np.atleast_3d(data.YBH*ones)
    ZBH   = np.atleast_3d(data.ZBH*ones)
    ZBHbv = np.atleast_3d(data.ZBH*ones)

    XA1   = np.atleast_3d(data.XA1*inv_root_beta)
    YA1   = np.atleast_3d(data.YA1*ones)
    ZA1   = np.atleast_3d(data.ZA1*ones)
    XA2   = np.atleast_3d(data.XA2*inv_root_beta)
    YA2   = np.atleast_3d(data.YA2*ones)
    ZA2   = np.atleast_3d(data.ZA2*ones)

    XB1   = np.atleast_3d(data.XB1*inv_root_beta)
    YB1   = np.atleast_3d(data.YB1*ones)
    ZB1   = np.atleast_3d(data.ZB1*ones)
    XB2   = np.atleast_3d(data.XB2*inv_root_beta)
    YB2   = np.atleast_3d(data.YB2*ones)
    ZB2   = np.atleast_3d(data.ZB2*ones)
          
    XAC   = np.atleast_3d(data.XAC*inv_root_beta)
    YAC   = np.atleast_3d(data.YAC*ones)
    ZAC   = np.atleast_3d(data.ZAC*ones)
    XBC   = np.atleast_3d(data.XBC*inv_root_beta)
    YBC   = np.atleast_3d(data.YBC*ones)
    ZBC   = np.atleast_3d(data.ZBC*ones)
    
    XA_TE   = np.atleast_3d(data.XA_TE*inv_root_beta)
    YA_TE   = np.atleast_3d(data.YA_TE*ones)
    ZA_TE   = np.atleast_3d(data.ZA_TE*ones)
    XB_TE   = np.atleast_3d(data.XB_TE*inv_root_beta)
    YB_TE   = np.atleast_3d(data.YB_TE*ones)
    ZB_TE   = np.atleast_3d(data.ZB_TE*ones) 
    
    XC    = np.atleast_3d(data.XC*inv_root_beta)
    YC    = np.atleast_3d(data.YC*ones) 
    ZC    = np.atleast_3d(data.ZC*ones)  
    n_w   = data.n_w

    theta_w = np.atleast_3d(theta_w)   # wake model, use theta_w if setting to freestream, use 0 if setting to airfoil chord like
    n_aoa   = np.shape(theta_w)[0]
    
    # -------------------------------------------------------------------------------------------
    # Compute velocity induced by horseshoe vortex segments on every control point by every panel
    # ------------------------------------------------------------------------------------------- 
    ## If YBH is negative, flip A and B, ie negative side of the airplane. Vortex order flips
    boolean = YBH<0.
    XA1[boolean], XB1[boolean] = XB1[boolean], XA1[boolean]
    YA1[boolean], YB1[boolean] = YB1[boolean], YA1[boolean]
    ZA1[boolean], ZB1[boolean] = ZB1[boolean], ZA1[boolean]
    XA2[boolean], XB2[boolean] = XB2[boolean], XA2[boolean]
    YA2[boolean], YB2[boolean] = YB2[boolean], YA2[boolean]
    ZA2[boolean], ZB2[boolean] = ZB2[boolean], ZA2[boolean]    
    XAH[boolean], XBH[boolean] = XBH[boolean], XAH[boolean]
    YAH[boolean], YBH[boolean] = YBH[boolean], YAH[boolean] 
    ZAH[boolean], ZBH[boolean] = ZBH[boolean], ZAH[boolean]

    XA_TE[boolean], XB_TE[boolean] = XB_TE[boolean], XA_TE[boolean]
    YA_TE[boolean], YB_TE[boolean] = YB_TE[boolean], YA_TE[boolean]
    ZA_TE[boolean], ZB_TE[boolean] = ZB_TE[boolean], ZA_TE[boolean]

    # Transpose thing
    XC = np.swapaxes(XC,1,2) 
    YC = np.swapaxes(YC,1,2) 
    ZC = np.swapaxes(ZC,1,2)  
    
    # compute influence of bound vortices 
    C_AB_bv = np.transpose(vortex(XC, YC, ZC, XAH, YAH, ZAH, XBH, YBH, ZBH),axes=[1,2,3,0])
    
    # compute influence of 3/4 left legs 
    C_AB_34_ll = np.transpose(vortex(XC, YC, ZC, XA2, YA2, ZA2, XAH, YAH, ZAH),axes=[1,2,3,0]) # original

    # compute influence of whole panel left legs  
    C_AB_ll   =  np.transpose(vortex(XC, YC, ZC, XA2, YA2, ZA2, XA1, YA1, ZA1),axes=[1,2,3,0]) # original

    # compute influence of 3/4 right legs  
    C_AB_34_rl = np.transpose(vortex(XC, YC, ZC, XBH, YBH, ZBH, XB2, YB2, ZB2),axes=[1,2,3,0]) # original 

    # compute influence of whole right legs   
    C_AB_rl = np.transpose(vortex(XC, YC, ZC, XB1, YB1, ZB1, XB2, YB2, ZB2),axes=[1,2,3,0]) # original 

    # velocity induced by left leg of vortex (A to inf)
    C_Ainf  = np.transpose(vortex_to_inf_l(XC, YC, ZC, XA_TE, YA_TE, ZA_TE,theta_w),axes=[1,2,3,0])

    # velocity induced by right leg of vortex (B to inf)
    C_Binf  = np.transpose(vortex_to_inf_r(XC, YC, ZC, XB_TE, YB_TE, ZB_TE,theta_w),axes=[1,2,3,0])

    # compute Mach Cone Matrix
    MCM = np.ones_like(C_AB_bv)
    MCM = compute_mach_cone_matrix(XC,YC,ZC,MCM,mach)
    data.MCM = MCM 
    n_cp     = n_w*n_cw*n_sw 
    # multiply by mach cone 
    C_AB_bv     = C_AB_bv    * MCM
    C_AB_34_ll  = C_AB_34_ll * MCM
    C_AB_ll     = C_AB_ll    * MCM
    C_AB_34_rl  = C_AB_34_rl * MCM
    C_AB_rl     = C_AB_rl    * MCM
    C_Ainf      = C_Ainf     * MCM
    C_Binf      = C_Binf     * MCM  
    
    # the follow block of text adds up all the trailing legs of the vortices which are on the wing for the downwind panels   
    C_AB_ll_on_wing = np.zeros_like(C_AB_ll)
    C_AB_rl_on_wing = np.zeros_like(C_AB_ll)

    for n in range(n_cp):
            n_te_p = (n_cw-(n+1)%n_cw)
            if (n+1)%n_cw != 0:
                start = n+1
                end   = n+n_te_p
                C_AB_ll_on_wing[:,:,n,:] = np.sum(C_AB_ll[:,:,start:end,:],axis=2) 
                C_AB_rl_on_wing[:,:,n,:] = np.sum(C_AB_rl[:,:,start:end,:],axis=2)                
    
    # Add all the influences together
    C_AB_ll_tot = C_AB_ll_on_wing + C_AB_34_ll + C_Ainf  # verified from book using example 7.4 pg 399-404
    C_AB_rl_tot = C_AB_rl_on_wing + C_AB_34_rl + C_Binf  # verified from book using example 7.4 pg 399-404
    C_mn        = C_AB_bv +  C_AB_ll_tot  + C_AB_rl_tot  # verified from book using example 7.4 pg 399-404
    
    DW_mn = 2*(C_AB_ll_tot  + C_AB_rl_tot) # summation of trailing vortices for semi infinite 
    
    return C_mn, DW_mn 

# -------------------------------------------------------------------------------
# vortex strength computation
# -------------------------------------------------------------------------------
def vortex(X,Y,Z,X1,Y1,Z1,X2,Y2,Z2):
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

    R1R2X  =   Y_Y1*Z_Z2 - Z_Z1*Y_Y2 
    R1R2Y  = -(X_X1*Z_Z2 - Z_Z1*X_X2)
    R1R2Z  =   X_X1*Y_Y2 - Y_Y1*X_X2
    SQUARE = R1R2X*R1R2X + R1R2Y*R1R2Y + R1R2Z*R1R2Z
    SQUARE[SQUARE==0] = 1e-32
    R1     = np.sqrt(X_X1*X_X1 + Y_Y1*Y_Y1 + Z_Z1*Z_Z1) 
    R2     = np.sqrt(X_X2*X_X2 + Y_Y2*Y_Y2 + Z_Z2*Z_Z2) 
    R0R1   = X2_X1*X_X1 + Y2_Y1*Y_Y1 + Z2_Z1*Z_Z1
    R0R2   = X2_X1*X_X2 + Y2_Y1*Y_Y2 + Z2_Z1*Z_Z2
    RVEC   = np.array([R1R2X,R1R2Y,R1R2Z])
    COEF   = (1/(4*np.pi))*(RVEC/SQUARE) * (R0R1/R1 - R0R2/R2)

    if np.isnan(COEF).any():
        print('NaN!')       

    return COEF

def vortex_to_inf_l(X,Y,Z,X1,Y1,Z1,tw): 

    # Take all the differences
    X_X1  = X-X1    
    Y_Y1  = Y-Y1
    Y1_Y  = Y1-Y
    Z_Z1  = Z-Z1

    DENUM =  Z_Z1*Z_Z1 + Y1_Y*Y1_Y    
    DENUM[DENUM==0] = 1e-32
    XVEC  = -Y1_Y*np.sin(tw)/DENUM
    YVEC  =  (Z_Z1)/DENUM
    ZVEC  =  Y1_Y*np.cos(tw)/DENUM
    BRAC  =  1 + (X_X1 / (np.sqrt(X_X1*X_X1 + Y_Y1*Y_Y1 + Z_Z1*Z_Z1)))
    RVEC   = np.array([XVEC, YVEC, ZVEC])
    COEF  = (1/(4*np.pi))*RVEC*BRAC  
    
    if np.isnan(COEF).any():
        print('NaN!')   
        
    return COEF

def vortex_to_inf_r(X,Y,Z,X1,Y1,Z1,tw):

    # Take all the differences
    X_X1  = X-X1    
    Y_Y1  = Y-Y1
    Y1_Y  = Y1-Y
    Z_Z1  = Z-Z1    

    DENUM =  Z_Z1*Z_Z1 + Y1_Y*Y1_Y
    DENUM[DENUM==0] = 1e-32
    XVEC  =  Y1_Y*np.sin(tw)/DENUM
    YVEC  = -Z_Z1/DENUM
    ZVEC  = -Y1_Y*np.cos(tw)/DENUM
    BRAC  =  1 + (X_X1 / (np.sqrt(X_X1*X_X1+ Y_Y1*Y_Y1+ Z_Z1*Z_Z1)))
    RVEC   = np.array([XVEC, YVEC, ZVEC])
    COEF  = (1/(4*np.pi))*RVEC*BRAC      
    
    if np.isnan(COEF).any():
        print('NaN!')   
        
    return COEF

def compute_mach_cone_matrix(XC,YC,ZC,MCM,mach):
    for m_idx in range(len(mach)):
        c = np.arcsin(1/mach[m_idx])
        for cp_idx in range(len(XC[m_idx,:])):
            del_x = XC[m_idx,:] - XC[m_idx,cp_idx] 
            del_y = YC[m_idx,:] - YC[m_idx,cp_idx] 
            del_z = ZC[m_idx,:] - ZC[m_idx,cp_idx] 
            flag  = -c*del_x**2 + del_y**2 + del_z**2
            idxs  = np.where(flag > 0.0)[0]
            MCM[m_idx,cp_idx,idxs]  = [0.0, 0.0, 0.0]     
    return MCM