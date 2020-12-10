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
def compute_wing_induced_velocity_sup(VD,n_sw,n_cw,theta_w,mach,use_MCM = False, grid_stretch_super = False):
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
    inv_root_beta = np.zeros_like(mach)
    mach[mach==1]         = 1.001  
    inv_root_beta[mach<1] = 1.
    inv_root_beta[mach<0.3] = 1.0
    inv_root_beta[mach>1]   = 1.0
    yz_stretch = ones*1.0
    
    if grid_stretch_super==False:
        inv_root_beta[mach>1] = 1.
    inv_root_beta = np.atleast_3d(inv_root_beta)
    
     
    XAH   = np.atleast_3d(VD.XAH*inv_root_beta) 
    YAH   = np.atleast_3d(VD.YAH*yz_stretch) 
    ZAH   = np.atleast_3d(VD.ZAH*yz_stretch) 
    XBH   = np.atleast_3d(VD.XBH*inv_root_beta) 
    YBH   = np.atleast_3d(VD.YBH*yz_stretch) 
    ZBH   = np.atleast_3d(VD.ZBH*yz_stretch) 

    XA1   = np.atleast_3d(VD.XA1*inv_root_beta)
    YA1   = np.atleast_3d(VD.YA1*yz_stretch)
    ZA1   = np.atleast_3d(VD.ZA1*yz_stretch)
    XA2   = np.atleast_3d(VD.XA2*inv_root_beta)
    YA2   = np.atleast_3d(VD.YA2*yz_stretch)
    ZA2   = np.atleast_3d(VD.ZA2*yz_stretch)

    XB1   = np.atleast_3d(VD.XB1*inv_root_beta)
    YB1   = np.atleast_3d(VD.YB1*yz_stretch)
    ZB1   = np.atleast_3d(VD.ZB1*yz_stretch)
    XB2   = np.atleast_3d(VD.XB2*inv_root_beta)
    YB2   = np.atleast_3d(VD.YB2*yz_stretch)
    ZB2   = np.atleast_3d(VD.ZB2*yz_stretch) 
    
    XC_TE   = np.atleast_3d(VD.XC_TE*inv_root_beta)
    YC_TE   = np.atleast_3d(VD.YC_TE*yz_stretch)
    ZC_TE   = np.atleast_3d(VD.ZC_TE*yz_stretch)    
    XA_TE   = np.atleast_3d(VD.XA_TE*inv_root_beta)
    YA_TE   = np.atleast_3d(VD.YA_TE*yz_stretch)
    ZA_TE   = np.atleast_3d(VD.ZA_TE*yz_stretch)
    XB_TE   = np.atleast_3d(VD.XB_TE*inv_root_beta)
    YB_TE   = np.atleast_3d(VD.YB_TE*yz_stretch)
    ZB_TE   = np.atleast_3d(VD.ZB_TE*yz_stretch) 
    
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
    XC    = np.swapaxes(XC,1,2) 
    YC    = np.swapaxes(YC,1,2) 
    ZC    = np.swapaxes(ZC,1,2)  
    XC_TE = np.swapaxes(XC_TE,1,2) 
    YC_TE = np.swapaxes(YC_TE,1,2) 
    ZC_TE = np.swapaxes(ZC_TE,1,2) 
    
    
    # These vortices will use AH and BH, rather than the typical location
    xa = XAH
    ya = YAH
    za = ZAH
    xb = XBH
    yb = YBH
    zb = ZBH
    
    # This is not the control point for the panel
    xc = 0.5*(xa+xb)
    yc = 0.5*(ya+yb)
    zc = 0.5*(za+zb)
    
    # This is the receiving point, or the control points
    xo = XC
    yo = YC
    zo = ZC
    
    theta = np.arctan2(zb-za,yb-ya)
    
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    
    # rotated axes
    x1bar = (xa - xc)
    y1bar = (ya - yc)*costheta + (za - zc)*sintheta
    z1bar =-(ya - yc)*sintheta + (za - zc)*costheta
    
    x2bar = (xb - xc)
    y2bar = (yb - yc)*costheta + (zb - zc)*sintheta
    z2bar =-(yb - yc)*sintheta + (zb - zc)*costheta
    
    xobar = (xo - xc)
    yobar = (yo - yc)*costheta + (zo - zc)*sintheta
    zobar =-(yo - yc)*sintheta + (zo - zc)*costheta
    
    #dimensions
    s = np.abs(y1bar)
    t = x1bar/y1bar
    
    x1 = xobar + t*s
    y1 = yobar + s
    
    x2 = xobar - t*s
    y2 = yobar - s
    
    xs = xobar -t*yobar

    
    # Calculate coefficients
    F1 = F(t, x1, beta_2, y1, zo)
    F2 = F(t, x2, beta_2, y2, zo)
    G1 = G(x1, beta_2, y1, zo)
    G2 = G(x2, beta_2, y2, zo)
    
    denom = bnd_vortex_denom(xs, t, beta_2, zo)
    
    U_rot = u(zo, denom, F1, F2)
    V_rot = v(F1, F2, t, G1, G2, denom, y1, y2, zo)
    W_rot = w(xs, F1, F2, denom, y1, y2, G1, G2, zo)
    
    U = (U_rot)/(2*np.pi*kappa)
    V = (V_rot*costheta - W_rot*sintheta)/(2*np.pi*kappa)
    W = (V_rot*sintheta + W_rot*costheta)/(2*np.pi*kappa)

    C_mn = np.zeros(np.shape(kappa)+(3,))
    C_mn[:,:,:,0] = U
    C_mn[:,:,:,1] = V
    C_mn[:,:,:,2] = W
    
    DW_mn = np.zeros_like(C_mn)



    return C_mn, DW_mn
    



def F(t,x,b2,y,z):
    
    denum = np.real(np.sqrt(x**2+b2*(y**2 + z**2)+0j))
    
    F = (t*x+b2*y)/denum
    
    F[F==np.inf]  = 0.
    F[F==-np.inf] = 0.
    
    return F

def G(x,b2,y,z):
    
    denum = np.real(np.sqrt(x**2 + (b2**2)*(y**2 + z**2) +0j))
    
    G = x/denum
    
    G[G==np.inf]  = 0.
    G[G==-np.inf] = 0.    
    
    G[b2>0] = G[b2>0] + 1
        
    return G

def bnd_vortex_denom(xs,t,b2,z):
    
    denom = xs**2 + (t**2+b2)*(z**2)

    return denom
    
    
def u(zo,denom,F1,F2):
    
    u = zo*(F1-F2)/denom
    
    return u

def v(F1,F2,t,G1,G2,denom,y1,y2,zo):
    
    v = zo*(-(F1-F2)*t/denom + G1/(y1**2+zo**2)-G2/(y2**2+zo**2))
    
    return v

def w(xs,F1,F2,denom,y1,y2,G1,G2,zo):
    
    w = -(xs*(F1-F2)/denom + y1*G1/(y1**2+zo**2) - y2*G2/(y2**2+zo**2))
    
    return w