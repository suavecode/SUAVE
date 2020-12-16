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
    
    
    # COMPUTE COORDINATES OF RECEIVING POINT WITH RESPECT TO END POINTS OF SKEWED LEG.
    s = np.abs(y1bar)
    t = x1bar/y1bar  
    length = np.shape(s)[-1]
    s = np.repeat(s,length,axis=1)
    t = np.repeat(t,length,axis=1)
    
    X1 = xobar + t*s
    Y1 = yobar + s
    X2 = xobar - t*s
    Y2 = yobar - s
    
    CUTOFF = 0.8
    
    
    # CALCULATE AXIAL DISTANCE BETWEEN PROJECTION OF RECEIVING POINT ONTO HORSESHOE PLANE AND EXTENSION OF SKEWED LEG.
    XTY = xobar - t*yobar
    
    # ZERO-OUT PERTURBATION VELOCITY COMPONENTS
    U = np.zeros_like(beta_2)
    V = np.zeros_like(beta_2)
    W = np.zeros_like(beta_2)
    
    # The notation in this method is flipped from the paper
    B2 = - beta_2
    
    
    # SET VALUES OF NUMERICAL TOLERANCE CONSTANTS.
    TOL    = s /500.0
    TOLSQ  = TOL *TOL
    TOLSQ2 = 2500.0 *TOLSQ
    ZSQ    = zobar *zobar
    YSQ1   = Y1 *Y1
    YSQ2   = Y2 *Y2
    RTV1   = YSQ1 + ZSQ
    RTV2   = YSQ2 + ZSQ
    RO1    = B2 *RTV1
    RO2    = B2 *RTV2
    RAD1   = np.zeros_like(beta_2)
    RAD2   = np.zeros_like(beta_2)
    XSQ1   = X1 *X1
    XSQ2   = X2 *X2
    
    # Split the vectors into subsonic and supersonic
    sub = beta_2>0
    sup = beta_2<=0
    
    B2_sub     = B2[sub]
    B2_sup     = B2[sup]
    XSQ1_sub   = XSQ1[sub]
    XSQ1_sup   = XSQ1[sup]
    XSQ2_sub   = XSQ2[sub]
    XSQ2_sup   = XSQ2[sup]
    RO1_sub    = RO1[sub]
    RO1_sup    = RO1[sup]
    RO2_sub    = RO2[sub]
    RO2_sup    = RO2[sup]    
    XTY_sub    = XTY[sub]
    XTY_sup    = XTY[sup]
    T_sub      = t[sub]
    T_sup      = t[sup]
    ZSQ_sub    = ZSQ[sub]
    ZSQ_sup    = ZSQ[sup]
    X1_sub     = X1[sub]
    X1_sup     = X1[sup]
    X2_sub     = X2[sub]
    X2_sup     = X2[sup]    
    Y1_sub     = Y1[sub]
    Y1_sup     = Y1[sup]
    Y2_sub     = Y2[sub]
    Y2_sup     = Y2[sup]
    RAD1_sub   = RAD1[sub]
    RAD1_sup   = RAD1[sup]
    RAD2_sub   = RAD2[sub]
    RAD2_sup   = RAD2[sup]
    RTV1_sub   = RTV1[sub]
    RTV1_sup   = RTV1[sup]
    RTV2_sub   = RTV2[sub]
    RTV2_sup   = RTV2[sup]
    TOL_sub    = TOL[sub]
    TOL_sup    = TOL[sup]
    TOLSQ_sub  = TOLSQ[sub]
    TOLSQ_sup  = TOLSQ[sup]
    TOLSQ2_sub = TOLSQ2[sub]
    TOLSQ2_sup = TOLSQ2[sup] 
    zobar_sub  = zobar[sub]
    zobar_sup  = zobar[sup]
    
    
    # COMPUTATION FOR SUBSONIC HORSESHOE VORTEX
    U_sub, V_sub, W_sub = subsonic(zobar_sub,XSQ1_sub,RO1_sub,XSQ2_sub,RO2_sub,XTY_sub,T_sub,B2_sub,ZSQ_sub,TOLSQ_sub,X1_sub,Y1_sub,X2_sub,Y2_sub,RAD1_sub,RAD2_sub,RTV1_sub,RTV2_sub)
    
    # Update the velocities
    U[sub] = U_sub
    V[sub] = V_sub
    W[sub] = W_sub
    
    # COMPUTATION FOR SUPERSONIC HORSESHOE VORTEX
    U_sup, V_sup, W_sup = supersonic(zobar_sup,XSQ1_sup,RO1_sup,XSQ2_sup,RO2_sup,XTY_sup,T_sup,B2_sup,ZSQ_sup,TOLSQ_sup,TOL_sup,TOLSQ2_sup,X1_sup,Y1_sup,X2_sup,Y2_sup,RAD1_sup,RAD2_sup,RTV1_sup,RTV2_sup,CUTOFF)
    
    # Update the velocities
    U[sup] = U_sup
    V[sup] = V_sup
    W[sup] = W_sup
    
    U_rot = U
    V_rot = V
    W_rot = W
    
    # Velocities in the vehicles frame
    U = (U_rot)
    V = (V_rot*costheta - W_rot*sintheta)
    W = (V_rot*sintheta + W_rot*costheta)
    
    
    # Pack into matrices
    C_mn = np.zeros(np.shape(kappa)+(3,))
    C_mn[:,:,:,0] = U
    C_mn[:,:,:,1] = V
    C_mn[:,:,:,2] = W

    DW_mn = np.zeros_like(C_mn)
    DW_mn[:,:,:,1] = V
    DW_mn[:,:,:,2] = W
    

    return C_mn, DW_mn    
    
    
def subsonic(Z,XSQ1,RO1,XSQ2,RO2,XTY,T,B2,ZSQ,TOLSQ,X1,Y1,X2,Y2,RAD1,RAD2,RTV1,RTV2):
    
    CPI  = 4 * np.pi
    ARG1 = XSQ1 - RO1
    RAD1 = np.sqrt(ARG1)
    ARG2 = XSQ2 - RO2
    RAD2 = np.sqrt(ARG2)
    
    XBSQ = XTY * XTY
    TBZ  = (T*T-B2)*ZSQ
    DENOM = XBSQ + TBZ
    
    DENOM[DENOM<TOLSQ] = TOLSQ[DENOM<TOLSQ]
    
    FB1 = (T *X1 - B2 *Y1) /RAD1
    FT1 = (X1 + RAD1) /(RAD1 *RTV1)
    FT1[RTV1<TOLSQ] = 0.
    
    FB2 = (T *X2 - B2 *Y2) /RAD2
    FT2 = (X2 + RAD2) /(RAD2 *RTV2)
    FT2[RTV2<TOLSQ] = 0.
    
    QB = (FB1 - FB2) /DENOM
    ZETAPI = Z /CPI
    U = ZETAPI *QB
    U[ZSQ<TOLSQ] = 0.
    V = ZETAPI * (FT1 - FT2 - QB *T)
    V[ZSQ<TOLSQ] = 0.
    W = - (QB *XTY + FT1 *Y1 - FT2 *Y2) /CPI
    
    return U, V, W

    
def supersonic(Z,XSQ1,RO1,XSQ2,RO2,XTY,T,B2,ZSQ,TOLSQ,TOL,TOLSQ2,X1,Y1,X2,Y2,RAD1,RAD2,RTV1,RTV2,CUTOFF):
    
    CPI  = 2 * np.pi
    ARG1 = XSQ1 - RO1
    ARG2 = XSQ2 - RO2
    
    RAD1[ARG1>0.] = np.sqrt(ARG1[ARG1>0.] )
    RAD2[ARG2>0.] = np.sqrt(ARG2[ARG2>0.])
    

    ZETAPI = Z/CPI
    
    XBSQ  = XTY * XTY
    TBZ   = (T *T - B2) *ZSQ
    DENOM = XBSQ + TBZ
    SIGN = np.ones_like(B2)
    SIGN[DENOM<0] = -1.
    DENOM[np.abs(DENOM)<TOLSQ] = SIGN[np.abs(DENOM)<TOLSQ]*TOLSQ[np.abs(DENOM)<TOLSQ]
    
    # Create a boolean for various conditions for F1 that goes to zero
    bool1 = np.ones_like(B2) * True
    bool1[X1<TOL]   = False
    bool1[RAD1==0.] = False
    
    REPS = CUTOFF*XSQ1
    FRAD = RAD1

    bool1[RO1>REPS] = False
    FB1 = (T*X1-B2*Y1)/FRAD
    FT1 = X1/(FRAD*RTV1)
    FT1[RTV1<TOLSQ] = 0.
    
    # Use the boolean to turn things off
    FB1[np.isnan(FB1)] = 1.
    FT1[np.isnan(FT1)] = 1.
    FB1 = FB1*bool1
    FT1 = FT1*bool1
    
    # Round 2
    # Create a boolean for various conditions for F2 that goes to zero
    bool2 = np.ones_like(B2) * True
    bool2[X2<TOL]   = False
    bool2[RAD2==0.] = False
    
    REPS = CUTOFF *XSQ2
    FRAD = RAD2    
    
    bool1[RO2>REPS] = False
    FB2 = (T *X2 - B2 *Y2) /FRAD
    FT2 = X2 /(FRAD *RTV2)
    FT2[RTV2<TOLSQ] = 0.
    
    # Use the boolean to turn things off
    FB2[np.isnan(FB2)] = 1.
    FT2[np.isnan(FT2)] = 1.
    FB2 = FB2*bool2
    FT2 = FT2*bool2
    
    QB = (FB1 - FB2) /DENOM
    U = ZETAPI *QB
    V = ZETAPI *(FT1 - FT2 - QB *T)
    W = - (QB *XTY + FT1 *Y1 - FT2 *Y2) /CPI    
    
    # COMPUTATION FOR SUPERSONIC HORSESHOE VORTEX WHEN RECEIVING POINT IS IN THE PLANE OF THE HORSESHOE
    RAD1_in = RAD1[ZSQ<TOLSQ2]
    RAD2_in = RAD2[ZSQ<TOLSQ2] 
    Y1_in   = Y1[ZSQ<TOLSQ2]
    Y2_in   = Y2[ZSQ<TOLSQ2]
    XTY_in  = XTY[ZSQ<TOLSQ2]
    TOL_in  = TOL[ZSQ<TOLSQ2]
    
    U_in, V_in, W_in = supersonic_in_plane(RAD1_in, RAD2_in, Y1_in, Y2_in, TOL_in, XTY_in, CPI)
    
    U[ZSQ<TOLSQ2] = U_in
    V[ZSQ<TOLSQ2] = V_in
    W[ZSQ<TOLSQ2] = W_in
    
    
    return U, V, W


def supersonic_in_plane(RAD1,RAD2,Y1,Y2,TOL,XTY,CPI):
    
    F1 = np.zeros_like(RAD1)
    F2 = np.zeros_like(RAD2)
    
    F1[np.abs(Y1)>TOL] = RAD1[np.abs(Y1)>TOL]/Y1[np.abs(Y1)>TOL]
    F2[np.abs(Y2)>TOL] = RAD2[np.abs(Y2)>TOL]/Y2[np.abs(Y2)>TOL]
    
    U = np.zeros_like(RAD1)
    V = np.zeros_like(RAD1)
    W = np.zeros_like(RAD1)
    W[np.abs(XTY)>TOL] = (-F1[np.abs(XTY)>TOL] + F2[np.abs(XTY)>TOL])/(XTY[np.abs(XTY)>TOL]*CPI)
    
    
    
    return U, V, W
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ## I ARBITRARILY SET THIS TOLERANCE:
    #tol = 0.1
    
    ##dimensions
    #s = np.abs(y1bar)
    #t = x1bar/y1bar
    
    #x1 = xobar + t*s
    #y1 = yobar + s
    
    #x2 = xobar - t*s
    #y2 = yobar - s
    
    #xs = xobar - t*yobar
        
    ## Calculate coefficients
    #F1, G1, cone1 = F_and_G(t, x1, beta_2, y1, zobar)
    #F2, G2, cone2 = F_and_G(t, x2, beta_2, y2, zobar)
    
    #cone = np.logical_or(cone1,cone2)

    #d1 = d(y1, zobar)
    #d2 = d(y2, zobar)
    
    #denom = bnd_vortex_denom(xs, t, beta_2, zobar)
    
    ## Velocities in the frame of the vortex
    #U_rot = u(zo, denom, F1, F2)
    #V_rot = v(F1, F2, t, G1, G2, denom, zobar,d1,d2)
    #W_rot = w(xs, F1, F2, denom, y1, y2, G1, G2, zobar,d1,d2)
    
    #boolean = np.logical_and(((zobar**2) < (tol**2)),(beta_2<0))
    
    #RAD1 = np.sqrt(x1**2+beta_2*(y1**2))
    #RAD2 = np.sqrt(x2**2+beta_2*(y2**2))
    
    #RAD1[np.isnan(RAD1)] = 0.
    #RAD2[np.isnan(RAD2)] = 0.
    
    #F1_new = RAD1/y1
    #F2_new = RAD2/y2
    
    #U_rot[boolean] = 0.
    #V_rot[boolean] = 0.
    #W_rot[boolean] = (-1/xs[boolean])*(F1_new[boolean]-F2_new[boolean])
    
    
    ##v_dw_rot = zobar*(G1/(y1**2+zobar**2) - G2/(y2**2+zobar**2))
    ##w_dw_rot = -(y1*G1/(y1**2+zobar**2) - y2*G2/(y2**2+zobar**2))

    ##w_dw_rot[beta_2<0] = ((-1/xs)*(cone1*np.sqrt(x1**2+beta_2*(y1**2))/y1-cone2*np.sqrt(x2**2+beta_2*(y2**2))/y2))[beta_2<0]
    ##v_dw_rot[beta_2<0]  = np.zeros_like(w_dw_rot)[beta_2<0]
    ##w_dw_rot[np.isnan(w_dw_rot)] = 0.
    
    ## Velocities in the vehicles frame
    #U = (U_rot)/(2*np.pi*kappa)
    #V = (V_rot*costheta - W_rot*sintheta)/(2*np.pi*kappa)
    #W = (V_rot*sintheta + W_rot*costheta)/(2*np.pi*kappa)
    
    ##v_dw = (v_dw_rot*costheta - w_dw_rot*sintheta)/(2*np.pi*kappa)
    ##w_dw = (v_dw_rot*sintheta + w_dw_rot*costheta)/(2*np.pi*kappa)
        
    ## Pack into matrices
    #C_mn = np.zeros(np.shape(kappa)+(3,))
    #C_mn[:,:,:,0] = U
    #C_mn[:,:,:,1] = V
    #C_mn[:,:,:,2] = W

    #DW_mn = np.zeros_like(C_mn)
    ##DW_mn[:,:,:,1] = v_dw
    ##DW_mn[:,:,:,2] = w_dw 
    #DW_mn[:,:,:,1] = V
    #DW_mn[:,:,:,2] = W
    

    #return C_mn, DW_mn

#def F_and_G(t,x,b2,y,z):
    
    ##c = 0.8
    
    ##cone = x**2 + b2*(y**2 + z**2)/c
    ##cone = np.heaviside(cone,0.)

    ##denum = np.sqrt(x**2 + b2*(y**2 + z**2))
    ##denum[cone==0] = np.inf
    
    ##f = (t*x + b2*y)/denum #FB
    
    ##g = x/denum #FT

    ### Adding 1 takes the trailing legs to infinity. Supersonically the legs shouldn't extend forever
    ##g[b2>0] = g[b2>0] + 1

    #return f, g, cone

#def bnd_vortex_denom(xs,t,b2,z):
    
    #denom = xs**2 + (t**2 + b2)*(z**2)

    #return denom

#def u(zo,denom,F1,F2):
    
    #u = zo*(F1-F2)/denom
    
    #return u

#def v(F1,F2,t,G1,G2,denom,z,d1,d2):
    
    #v = z*(-(F1-F2)*t/denom + G1/d1 - G2/d2)
    
    #return v

#def w(xs,F1,F2,denom,y1,y2,G1,G2,z,d1,d2):
    
    #w = -(xs*(F1-F2)/denom + y1*G1/d1 - y2*G2/d2)
    
    #return w


#def d(y,z):
    
    
    #return y**2+z**2