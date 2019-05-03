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
def compute_induced_velocity_matrix(data,n_sw,n_cw,theta_w):     
    # unpack 
    XAH  = np.atleast_2d(data.XAH)
    YAH  = np.atleast_2d(data.YAH)
    ZAH  = np.atleast_2d(data.ZAH)
    XBH  = np.atleast_2d(data.XBH)
    YBH  = np.atleast_2d(data.YBH)
    ZBH  = np.atleast_2d(data.ZBH)

    XA1  = np.atleast_2d(data.XA1)
    YA1  = np.atleast_2d(data.YA1)
    ZA1  = np.atleast_2d(data.ZA1)
    XA2  = np.atleast_2d(data.XA2)
    YA2  = np.atleast_2d(data.YA2)
    ZA2  = np.atleast_2d(data.ZA2)
          
    XB1  = np.atleast_2d(data.XB1)
    YB1  = np.atleast_2d(data.YB1)
    ZB1  = np.atleast_2d(data.ZB1)
    XB2  = np.atleast_2d(data.XB2)
    YB2  = np.atleast_2d(data.YB2)
    ZB2  = np.atleast_2d(data.ZB2)
          
    XAC  = np.atleast_2d(data.XAC)
    YAC  = np.atleast_2d(data.YAC)
    ZAC  = np.atleast_2d(data.ZAC)
    XBC  = np.atleast_2d(data.XBC)
    YBC  = np.atleast_2d(data.YBC)
    ZBC  = np.atleast_2d(data.ZBC)
    XC   = np.atleast_2d(data.XC)
    YC   = np.atleast_2d(data.YC) 
    ZC   = np.atleast_2d(data.ZC)  
    n_w  = data.n_w
    
    # ---------------------------------------------------------------------------------------
    # Compute velocity induced by horseshoe vortex segments on every control point by every panel
    # --------------------------------------------------------------------------------------- 
    n_cp     = n_w*n_cw*n_sw # total number of control points 

    # Make all the hats
    n_cw_1   = n_cw-1
    XC_hats  = np.atleast_2d(np.cos(theta_w)*XC + np.sin(theta_w)*ZC)
    YC_hats  = np.atleast_2d(YC)
    ZC_hats  = np.atleast_2d(-np.sin(theta_w)*XC + np.cos(theta_w)*ZC)

    XA2_hats = np.atleast_2d(np.cos(theta_w)*XA2[0,n_cw_1::n_cw]   + np.sin(theta_w)*ZA2[0,n_cw_1::n_cw])
    YA2_hats = np.atleast_2d(YA2[0,n_cw_1::n_cw])
    ZA2_hats = np.atleast_2d(-np.sin(theta_w)*XA2[0,n_cw_1::n_cw]  + np.cos(theta_w)*ZA2[0,n_cw_1::n_cw])
    
    XB2_hats = np.atleast_2d(np.cos(theta_w)*XB2[0,n_cw_1::n_cw]   + np.sin(theta_w)*ZB2[0,n_cw_1::n_cw])
    YB2_hats = np.atleast_2d(YB2[0,n_cw_1::n_cw]) 
    ZB2_hats = np.atleast_2d(-np.sin(theta_w)*XB2[0,n_cw_1::n_cw]  + np.cos(theta_w)*ZB2[0,n_cw_1::n_cw])
    
    # Expand out the hats to be of length n instead of n_sw
    XA2_hats = np.atleast_2d(np.repeat(XA2_hats,n_cw))
    YA2_hats = np.atleast_2d(np.repeat(YA2_hats,n_cw))
    ZA2_hats = np.atleast_2d(np.repeat(ZA2_hats,n_cw))
    XB2_hats = np.atleast_2d(np.repeat(XB2_hats,n_cw))
    YB2_hats = np.atleast_2d(np.repeat(YB2_hats,n_cw))
    ZB2_hats = np.atleast_2d(np.repeat(ZB2_hats,n_cw))
    
    # If YBH is negative, flip A and B, ie negative side of the airplane. Vortex order flips
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
    
    XA2_hats[boolean], XB2_hats[boolean] = XB2_hats[boolean], XA2_hats[boolean]
    YA2_hats[boolean], YB2_hats[boolean] = YB2_hats[boolean], YA2_hats[boolean]
    ZA2_hats[boolean], ZB2_hats[boolean] = ZB2_hats[boolean], ZA2_hats[boolean]         

    # compute influence of bound vortices 
    C_AB_bv = vortex(XC, YC, ZC, XAH.T, YAH.T, ZAH.T, XBH.T, YBH.T, ZBH.T).T

    # compute influence of 3/4 left legs
    C_AB_34_ll = vortex(XC, YC, ZC, XA2.T, YA2.T, ZA2.T, XAH.T, YAH.T, ZAH.T).T

    # compute influence of whole panel left legs 
    C_AB_ll   =  vortex(XC, YC, ZC, XA2.T, YA2.T, ZA2.T, XA1.T, YA1.T, ZA1.T).T

    # compute influence of 3/4 right legs
    C_AB_34_rl = vortex(XC, YC, ZC, XBH.T, YBH.T, ZBH.T, XB2.T, YB2.T, ZB2.T).T

    # compute influence of whole right legs 
    C_AB_rl = vortex(XC, YC, ZC, XB1.T, YB1.T, ZB1.T, XB2.T, YB2.T, ZB2.T).T

    # velocity induced by left leg of vortex (A to inf)
    C_Ainf  = vortex_to_inf_l(XC_hats, YC_hats, ZC_hats, XA2_hats.T, YA2_hats.T, ZA2_hats.T,theta_w).T

    # velocity induced by right leg of vortex (B to inf)
    C_Binf  = vortex_to_inf_r(XC_hats, YC_hats, ZC_hats, XB2_hats.T, YB2_hats.T, ZB2_hats.T,theta_w).T
                
    # Add the right and left influences seperateky
    C_AB_llrl_roll = C_AB_ll+C_AB_rl
    
    # Prime the arrays
    mask      = np.ones((n_cp,n_cp,3))
    C_AB_llrl = np.zeros((n_cp,n_cp,3))

    for idx in range(n_cw-1):    
        
        # Make a mask
        mask[:,n_cw-idx-1::n_cw,:]= np.zeros_like(mask[:,n_cw-idx-1::n_cw,:])  
        
        # Roll it over to the next component
        C_AB_llrl_roll = np.roll(C_AB_llrl_roll,-1,axis=1)   
        
        # Add in the components that we need
        C_AB_llrl = C_AB_llrl+ C_AB_llrl_roll*mask

    # Add all the influences together
    C_mn = C_AB_34_ll + C_AB_bv + C_AB_34_rl + C_Ainf + C_Binf + C_AB_llrl
    
    return C_mn

# -------------------------------------------------------------------------------
# vortex strength computation
# -------------------------------------------------------------------------------
def vortex(X,Y,Z,X1,Y1,Z1,X2,Y2,Z2):
    R1R2X  =   (Y-Y1)*(Z-Z2) - (Z-Z1)*(Y-Y2) 
    R1R2Y  = -((X-X1)*(Z-Z2) - (Z-Z1)*(X-X2))
    R1R2Z  =   (X-X1)*(Y-Y2) - (Y-Y1)*(X-X2)
    SQUARE = R1R2X**2 + R1R2Y**2 + R1R2Z**2
    R1     = np.sqrt((X-X1)**2 + (Y-Y1)**2 + (Z-Z1)**2)
    R2     = np.sqrt((X-X2)**2 + (Y-Y2)**2 + (Z-Z2)**2)
    R0R1   = (X2-X1)*(X-X1) + (Y2-Y1)*(Y-Y1) + (Z2-Z1)*(Z-Z1)
    R0R2   = (X2-X1)*(X-X2) + (Y2-Y1)*(Y-Y2) + (Z2-Z1)*(Z-Z2)
    RVEC   = np.array([R1R2X,R1R2Y,R1R2Z])
    COEF   = (1/(4*np.pi))*(RVEC/SQUARE) * (R0R1/R1 - R0R2/R2)
    return COEF

def vortex_to_inf_l(X,Y,Z,X1,Y1,Z1,tw): 
    DENUM =  (Z-Z1)**2 + (Y1-Y)**2
    XVEC  = -(Y1-Y)*np.sin(tw)/DENUM
    YVEC  =  (Z-Z1)/DENUM
    ZVEC  =  (Y1-Y)*np.cos(tw)/DENUM
    BRAC  =  1 + ((X-X1) / (np.sqrt((X-X1)**2+ (Y-Y1)**2+ (Z-Z1)**2)))
    RVEC   = np.array([XVEC, YVEC, ZVEC])
    COEF  = (1/(4*np.pi))*RVEC*BRAC         
    return COEF

def vortex_to_inf_r(X,Y,Z,X1,Y1,Z1,tw):
    DENUM =  (Z-Z1)**2 + (Y1-Y)**2
    XVEC  = (Y1-Y)*np.sin(tw)/DENUM
    YVEC  = -(Z-Z1)/DENUM
    ZVEC  = -(Y1-Y)*np.cos(tw)/DENUM
    BRAC  =  1 + ((X-X1) / (np.sqrt((X-X1)**2+ (Y-Y1)**2+ (Z-Z1)**2)))
    RVEC   = np.array([XVEC, YVEC, ZVEC])
    COEF  = (1/(4*np.pi))*RVEC*BRAC         
    return COEF