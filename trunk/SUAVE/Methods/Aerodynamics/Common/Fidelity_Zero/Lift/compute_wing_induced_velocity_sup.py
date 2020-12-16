## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_wing_induced_velocity.py
# 
# Created:  May 2018, M. Clarke
#           Apr 2020, M. Clarke
#           Jun 2020, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
import numpy as np 

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_wing_induced_velocity_sup(VD,n_sw,n_cw,theta_w,mach):
    """ This computes the induced velocitys are each control point 
    of the vehicle vortex lattice 

    Assumptions: 
    Trailing vortex legs infinity are alligned to freestream

    Source:  
    None

    Inputs: 
    VD       - vehicle vortex distribution      [Unitless] 
    n_sw     - number_spanwise_vortices         [Unitless]
    n_cw     - number_chordwise_vortices        [Unitless] 
    mach                                        [Unitless] 
    theta_w  - freestream wake angle            [radians]
    
    Outputs:                                
    C_mn     - total induced velocity matrix    [Unitless] 
    DW_mn    - induced downwash velocity matrix [Unitless] 

    Properties Used:
    N/A
    """
    # unpack  
    ones     = np.atleast_3d(np.ones_like(theta_w))
 
    # Prandtl Glauert Transformation for subsonic
    inv_root_beta = np.ones_like(mach)
    mach[mach==1]         = 1.001  
    inv_root_beta[mach<1] = 1.
    inv_root_beta[mach<0.3] = 1.0
    inv_root_beta[mach>1]   = 1.0
    yz_stretch = np.ones_like(mach)
    
    yz_stretch = np.atleast_3d(yz_stretch)
    inv_root_beta = np.atleast_3d(inv_root_beta)
    
    # Control points from the VLM 
    XAH   = np.atleast_3d(VD.XAH*inv_root_beta) 
    YAH   = np.atleast_3d(VD.YAH*yz_stretch) 
    ZAH   = np.atleast_3d(VD.ZAH*yz_stretch) 
    XBH   = np.atleast_3d(VD.XBH*inv_root_beta) 
    YBH   = np.atleast_3d(VD.YBH*yz_stretch) 
    ZBH   = np.atleast_3d(VD.ZBH*yz_stretch) 
    XC    = np.atleast_3d(VD.XC*inv_root_beta)
    YC    = np.atleast_3d(VD.YC*yz_stretch) 
    ZC    = np.atleast_3d(VD.ZC*yz_stretch)  
    
    # supersonic corrections
    kappa = np.ones_like(XAH)
    kappa[mach<1.,:] = 2.
    beta_2 = 1-mach**2
    sized_ones = np.ones((np.shape(mach)[0],np.shape(XAH)[-1],np.shape(XAH)[-1]))
    beta_2 = np.atleast_3d(beta_2)
    beta_2 = beta_2*sized_ones
    kappa  = kappa*sized_ones

    theta_w = np.atleast_3d(theta_w)   # wake model, use theta_w if setting to freestream, use 0 if setting to airfoil chord like
    
    # -------------------------------------------------------------------------------------------
    # Compute velocity induced by horseshoe vortex segments on every control point by every panel
    # ------------------------------------------------------------------------------------------- 
    ## If YBH is negative, flip A and B, ie negative side of the airplane. Vortex order flips
    boolean = YBH<0. 
    XAH[boolean], XBH[boolean] = XBH[boolean], XAH[boolean]
    YAH[boolean], YBH[boolean] = YBH[boolean], YAH[boolean] 
    ZAH[boolean], ZBH[boolean] = ZBH[boolean], ZAH[boolean]
    
    # Transpose thing
    XC    = np.swapaxes(XC,1,2) 
    YC    = np.swapaxes(YC,1,2) 
    ZC    = np.swapaxes(ZC,1,2)  
    
    # These vortices will use AH and BH, rather than the typical location
    xa = XAH
    ya = YAH
    za = ZAH
    xb = XBH
    yb = YBH
    zb = ZBH
    
    # This is not the control point for the panel, its the middle front of the vortex
    xc = 0.5*(xa+xb)
    yc = 0.5*(ya+yb)
    zc = 0.5*(za+zb)
    
    # This is the receiving point, or the control points
    xo = XC
    yo = YC
    zo = ZC
    
    # Incline the vortex
    theta = np.arctan2(zb-za,yb-ya)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    
    # rotated axes
    x1bar = (xa - xc)
    y1bar = (ya - yc)*costheta + (za - zc)*sintheta
    xobar = (xo - xc)
    yobar = (yo - yc)*costheta + (zo - zc)*sintheta
    zobar =-(yo - yc)*sintheta + (zo - zc)*costheta
    
    
    
    # I ARBITRARILY SET THIS TOLERANCE:
    tol = 0.1
    
    #dimensions
    s = np.abs(y1bar)
    t = x1bar/y1bar
    
    x1 = xobar + t*s
    y1 = yobar + s
    
    x2 = xobar - t*s
    y2 = yobar - s
    
    xs = xobar - t*yobar
        
    # Calculate coefficients
    F1, G1, cone1 = F_and_G(t, x1, beta_2, y1, zobar)
    F2, G2, cone2 = F_and_G(t, x2, beta_2, y2, zobar)
    
    cone = np.logical_or(cone1,cone2)

    d1 = d(y1, zobar)
    d2 = d(y2, zobar)
    
    denom = bnd_vortex_denom(xs, t, beta_2, zobar)
    
    # Velocities in the frame of the vortex
    U_rot = u(zo, denom, F1, F2)
    V_rot = v(F1, F2, t, G1, G2, denom, zobar,d1,d2)
    W_rot = w(xs, F1, F2, denom, y1, y2, G1, G2, zobar,d1,d2)
    
    boolean = np.logical_and(((zobar**2) < (tol**2)),(beta_2<0))
    
    RAD1 = np.sqrt(x1**2+beta_2*(y1**2))
    RAD2 = np.sqrt(x2**2+beta_2*(y2**2))
    
    RAD1[np.isnan(RAD1)] = 0.
    RAD2[np.isnan(RAD2)] = 0.
    
    F1_new = RAD1/y1
    F2_new = RAD2/y2
    
    U_rot[boolean] = 0.
    V_rot[boolean] = 0.
    W_rot[boolean] = (-1/xs[boolean])*(F1_new[boolean]-F2_new[boolean])
    
    
    #v_dw_rot = zobar*(G1/(y1**2+zobar**2) - G2/(y2**2+zobar**2))
    #w_dw_rot = -(y1*G1/(y1**2+zobar**2) - y2*G2/(y2**2+zobar**2))

    #w_dw_rot[beta_2<0] = ((-1/xs)*(cone1*np.sqrt(x1**2+beta_2*(y1**2))/y1-cone2*np.sqrt(x2**2+beta_2*(y2**2))/y2))[beta_2<0]
    #v_dw_rot[beta_2<0]  = np.zeros_like(w_dw_rot)[beta_2<0]
    #w_dw_rot[np.isnan(w_dw_rot)] = 0.
    
    # Velocities in the vehicles frame
    U = (U_rot)/(2*np.pi*kappa)
    V = (V_rot*costheta - W_rot*sintheta)/(2*np.pi*kappa)
    W = (V_rot*sintheta + W_rot*costheta)/(2*np.pi*kappa)
    
    #v_dw = (v_dw_rot*costheta - w_dw_rot*sintheta)/(2*np.pi*kappa)
    #w_dw = (v_dw_rot*sintheta + w_dw_rot*costheta)/(2*np.pi*kappa)
        
    # Pack into matrices
    C_mn = np.zeros(np.shape(kappa)+(3,))
    C_mn[:,:,:,0] = U
    C_mn[:,:,:,1] = V
    C_mn[:,:,:,2] = W

    DW_mn = np.zeros_like(C_mn)
    #DW_mn[:,:,:,1] = v_dw
    #DW_mn[:,:,:,2] = w_dw 
    DW_mn[:,:,:,1] = V
    DW_mn[:,:,:,2] = W
    

    return C_mn, DW_mn

def F_and_G(t,x,b2,y,z):
    
    c = 0.8
    
    cone = x**2 + b2*(y**2 + z**2)/c
    cone = np.heaviside(cone,0.)

    denum = np.sqrt(x**2 + b2*(y**2 + z**2))
    denum[cone==0] = np.inf
    
    f = (t*x + b2*y)/denum #FB
    
    g = x/denum #FT

    # Adding 1 takes the trailing legs to infinity. Supersonically the legs shouldn't extend forever
    g[b2>0] = g[b2>0] + 1

    return f, g, cone

def bnd_vortex_denom(xs,t,b2,z):
    
    denom = xs**2 + (t**2 + b2)*(z**2)

    return denom

def u(zo,denom,F1,F2):
    
    u = zo*(F1-F2)/denom
    
    return u

def v(F1,F2,t,G1,G2,denom,z,d1,d2):
    
    v = z*(-(F1-F2)*t/denom + G1/d1 - G2/d2)
    
    return v

def w(xs,F1,F2,denom,y1,y2,G1,G2,z,d1,d2):
    
    w = -(xs*(F1-F2)/denom + y1*G1/d1 - y2*G2/d2)
    
    return w


def d(y,z):
    
    
    return y**2+z**2