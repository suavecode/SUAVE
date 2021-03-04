## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_wing_induced_velocity.py
# 
# Created:  Dec 2020, E. Botero
#

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
import numpy as np 

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_wing_induced_velocity(VD,n_sw,n_cw,theta_w,mach):
    """ This computes the induced velocities at each control point 
    of the vehicle vortex lattice 

    Assumptions: 
    Trailing vortex legs infinity are alligned to freestream

    Source:  
    1. Miranda, Luis R., Robert D. Elliot, and William M. Baker. "A generalized vortex 
    lattice method for subsonic and supersonic flow applications." (1977). (NASA CR)
    
    2. VORLAX Source Code

    Inputs: 
    VD       - vehicle vortex distribution                    [Unitless] 
    n_sw     - number_spanwise_vortices                       [Unitless]
    n_cw     - number_chordwise_vortices                      [Unitless] 
    mach                                                      [Unitless] 
    
    Outputs:                                
    C_mn     - total induced velocity matrix                  [Unitless] 
    s        - semispan of the horshoe vortex                 [m] 
    t        - tangent of the horshoe vortex                  [-] 
    CHORD    - chord length for a panel                       [m] 
    RFLAG    - sonic vortex flag                              [boolean] 
    ZETA     - tangent incidence angle of the chordwise strip [-] 

    Properties Used:
    N/A
    """
    # unpack  
    n_cp     = n_sw*n_cw
    n_w      = VD.n_w

    # Control points from the VLM 
    expansion = np.atleast_3d(np.ones_like(mach))
    XAH   = np.atleast_3d(VD.XAH*expansion) 
    YAH   = np.atleast_3d(VD.YAH*expansion) 
    ZAH   = np.atleast_3d(VD.ZAH*expansion) 
    XBH   = np.atleast_3d(VD.XBH*expansion) 
    YBH   = np.atleast_3d(VD.YBH*expansion) 
    ZBH   = np.atleast_3d(VD.ZBH*expansion) 
    XA1   = np.atleast_3d(VD.XA1*expansion) 
    YA1   = np.atleast_3d(VD.YA1*expansion) 
    ZA1   = np.atleast_3d(VD.ZA1*expansion) 
    XB1   = np.atleast_3d(VD.XB1*expansion) 
    YB1   = np.atleast_3d(VD.YB1*expansion) 
    ZB1   = np.atleast_3d(VD.ZB1*expansion)    
    XA2   = np.atleast_3d(VD.XA2*expansion) 
    YA2   = np.atleast_3d(VD.YA2*expansion) 
    ZA2   = np.atleast_3d(VD.ZA2*expansion) 
    XB2   = np.atleast_3d(VD.XB2*expansion) 
    YB2   = np.atleast_3d(VD.YB2*expansion) 
    ZB2   = np.atleast_3d(VD.ZB2*expansion)       
    XC    = np.atleast_3d(VD.XC*expansion)
    YC    = np.atleast_3d(VD.YC*expansion) 
    ZC    = np.atleast_3d(VD.ZC*expansion)  
    XA_TE = np.atleast_3d(VD.XA_TE*expansion)  
    XB_TE = np.atleast_3d(VD.XB_TE*expansion)  
    ZA_TE = np.atleast_3d(VD.ZA_TE*expansion)  
    ZB_TE = np.atleast_3d(VD.ZB_TE*expansion)      
    
    # supersonic corrections
    beta_2 = 1-mach**2
    sized_ones = np.ones((np.shape(mach)[0],np.shape(XAH)[-1],np.shape(XAH)[-1]))
    beta_2 = np.atleast_3d(beta_2)
    beta_2 = beta_2*sized_ones
    
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
    x1bar = (xb - xc)
    y1bar = (yb - yc)*costheta + (zb - zc)*sintheta
    
    xobar = (xo - xc)
    yobar = (yo - yc)*costheta + (zo - zc)*sintheta
    zobar =-(yo - yc)*sintheta + (zo - zc)*costheta
    
    # COMPUTE COORDINATES OF RECEIVING POINT WITH RESPECT TO END POINTS OF SKEWED LEG.
    s = np.abs(y1bar)
    t = x1bar/y1bar  
    length = np.shape(s)[-1]
    s = np.repeat(s,length,axis=1)
    t = np.repeat(t,length,axis=1)
    
    X1 = xobar + t*s # In a planar case XC-XAH
    Y1 = yobar + s   # In a planar case YC-YAH
    X2 = xobar - t*s # In a planar case XC-XBH
    Y2 = yobar - s   # In a planar case YC-YBH
    
    # The cutoff hardcoded into vorlax
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
    RFLAG  = np.ones((len(mach),n_cp*n_w))
    mach_f = np.broadcast_to(mach,np.shape(RFLAG))
    
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
    RAD1_sup   = RAD1[sup]
    RAD2_sup   = RAD2[sup]
    RTV1_sub   = RTV1[sub]
    RTV1_sup   = RTV1[sup]
    RTV2_sub   = RTV2[sub]
    RTV2_sup   = RTV2[sup]
    TOL_sup    = TOL[sup]
    TOLSQ_sub  = TOLSQ[sub]
    TOLSQ_sup  = TOLSQ[sup]
    TOLSQ2_sup = TOLSQ2[sup] 
    zobar_sub  = zobar[sub]
    zobar_sup  = zobar[sup]
    RFLAG_sup  = RFLAG[mach_f>1]
    
    if np.sum(sub)>0:
        # COMPUTATION FOR SUBSONIC HORSESHOE VORTEX
        U_sub, V_sub, W_sub = subsonic(zobar_sub,XSQ1_sub,RO1_sub,XSQ2_sub,RO2_sub,XTY_sub,T_sub,B2_sub,ZSQ_sub,\
                                       TOLSQ_sub,X1_sub,Y1_sub,X2_sub,Y2_sub,RTV1_sub,RTV2_sub)
    else:
        U_sub = []
        V_sub = []
        W_sub = []
  
    # Update the velocities
    U[sub] = U_sub
    V[sub] = V_sub
    W[sub] = W_sub
    
    # COMPUTATION FOR SUPERSONIC HORSESHOE VORTEX
    RNMAX       = n_cw # number of chordwise panels
    LE_A_pts_x  = XA1[:,:,0:n_cp*n_w:n_cw]
    LE_B_pts_x  = XB1[:,:,0:n_cp*n_w:n_cw]
    LE_X        = (LE_A_pts_x+LE_B_pts_x)/2
    LE_X        = np.repeat(LE_X,n_cw,axis=2)
    LE_A_pts_z  = ZA1[:,:,0:n_cp*n_w:n_cw]
    LE_B_pts_z  = ZB1[:,:,0:n_cp*n_w:n_cw]
    LE_Z        = (LE_A_pts_z+LE_B_pts_z)/2    
    LE_Z        = np.repeat(LE_Z,n_cw,axis=2)
    TE_X        = (XB_TE + XA_TE)/2
    TE_Z        = (ZB_TE + ZA_TE)/2
    CHORD       = np.sqrt((TE_X-LE_X)**2 + (TE_Z-LE_Z)**2 )
    CHORD       = np.repeat(CHORD,length,axis=1)
    EYE         = np.eye(np.shape(CHORD)[-1]).flatten()
    EYE         = np.tile(EYE,np.sum(mach>1))
    CHORD_sup   = CHORD[sup]
    ZETA        = (LE_Z-TE_Z)/(LE_X-TE_X) # Zeta is the tangent incidence angle of the chordwise strip. LE to TE
    ZETA        = ZETA[:,0,:] # Fix the shape for later
    
    if np.sum(sup)>0:
        U_sup, V_sup, W_sup, RFLAG_sup = supersonic(zobar_sup,XSQ1_sup,RO1_sup,XSQ2_sup,RO2_sup,XTY_sup,T_sup,B2_sup,\
                                                    ZSQ_sup,TOLSQ_sup,TOL_sup,TOLSQ2_sup,X1_sup,Y1_sup,X2_sup,Y2_sup,\
                                                    RAD1_sup,RAD2_sup,RTV1_sup,RTV2_sup,CUTOFF,CHORD_sup,RNMAX,\
                                                    EYE,n_cw,n_cp,n_w,RFLAG_sup)
    else:
        U_sup = []
        V_sup = []
        W_sup = []
    
    # Update the velocities
    U[sup] = U_sup
    V[sup] = V_sup
    W[sup] = W_sup
    
    RFLAG[mach_f>1] = RFLAG_sup    
    
    U_rot = U
    V_rot = V
    W_rot = W
    
    # Velocities in the vehicles frame
    U = (U_rot)
    V = (V_rot*costheta - W_rot*sintheta)
    W = (V_rot*sintheta + W_rot*costheta)
    
    # Pack into matrices
    C_mn = np.zeros(np.shape(beta_2)+(3,))
    C_mn[:,:,:,0] = U
    C_mn[:,:,:,1] = V
    C_mn[:,:,:,2] = W
    
    return C_mn, s, t, CHORD, RFLAG, ZETA
    
    
def subsonic(Z,XSQ1,RO1,XSQ2,RO2,XTY,T,B2,ZSQ,TOLSQ,X1,Y1,X2,Y2,RTV1,RTV2):
    """  This computes the induced velocities at each control point 
    of the vehicle vortex lattice for subsonic mach numbers

    Assumptions: 
    Trailing vortex legs infinity are alligned to freestream

    Source:  
    1. Miranda, Luis R., Robert D. Elliot, and William M. Baker. "A generalized vortex 
    lattice method for subsonic and supersonic flow applications." (1977). (NASA CR)
    
    2. VORLAX Source Code

    Inputs: 
    Z       Z relative location of the vortices          [m]
    XSQ1    X1 squared                                   [m^2]
    RO1     coefficient                                  [-]
    XSQ2    X2 squared                                   [m^2]
    RO2     coefficient                                  [-]
    XTY     AXIAL DISTANCE BETWEEN PROJECTION OF RECEIVING POINT ONTO HORSESHOE PLANE AND EXTENSION OF SKEWED LEG [m]
    T       tangent of the horshoe vortex                [-] 
    B2      mach^2-1 (-beta2)                            [-] 
    ZSQ     Z squared                                    [m^2] 
    TOLSQ   coefficient                                  [-]
    X1      X coordinate of the left side of the vortex  [m]
    Y1      Y coordinate of the left side of the vortex  [m]
    X2      X coordinate of the right side of the vortex [m]
    Y2      Y coordinate of the right side of the vortex [m]
    RTV1    coefficient                                  [-]
    RTV2    coefficient                                  [-]

    
    Outputs:           
    U       X velocity       [unitless]
    V       Y velocity       [unitless]
    W       Z velocity       [unitless]

    Properties Used:
    N/A
    """  
    
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

    
def supersonic(Z,XSQ1,RO1,XSQ2,RO2,XTY,T,B2,ZSQ,TOLSQ,TOL,TOLSQ2,X1,Y1,X2,Y2,RAD1,RAD2,RTV1,RTV2,CUTOFF,CHORD,RNMAX,EYE,n_cw,n_cp,n_w,RFLAG):
    """  This computes the induced velocities at each control point 
    of the vehicle vortex lattice for supersonic mach numbers

    Assumptions: 
    Trailing vortex legs infinity are alligned to freestream

    Source:  
    1. Miranda, Luis R., Robert D. Elliot, and William M. Baker. "A generalized vortex 
    lattice method for subsonic and supersonic flow applications." (1977). (NASA CR)
    
    2. VORLAX Source Code

    Inputs: 
    Z       Z relative location of the vortices          [m]
    XSQ1    X1 squared                                   [m^2]
    RO1     coefficient                                  [-]
    XSQ2    X2 squared                                   [m^2]
    RO2     coefficient                                  [-]
    XTY     AXIAL DISTANCE BETWEEN PROJECTION OF RECEIVING POINT ONTO HORSESHOE PLANE AND EXTENSION OF SKEWED LEG [m]
    T       tangent of the horshoe vortex                [-] 
    B2      mach^2-1 (-beta2)                            [-] 
    ZSQ     Z squared                                    [m^2] 
    TOLSQ   coefficient                                  [-]
    X1      X coordinate of the left side of the vortex  [m]
    Y1      Y coordinate of the left side of the vortex  [m]
    X2      X coordinate of the right side of the vortex [m]
    Y2      Y coordinate of the right side of the vortex [m]
    RAD1    array of zeros                               [-]
    RAD2    array of zeros                               [-]
    RTV1    coefficient                                  [-]
    RTV2    coefficient                                  [-]
    CUTOFF  coefficient                                  [-]
    CHORD   chord length for a panel                     [m] 
    RNMAX   number of chordwise panels                   [-]
    EYE     eye matrix (linear algebra)                  [-]
    n_cw    number of chordwise panels                   [-]
    n_cp    number of control points                     [-]
    n_w     number of wings                              [-]
    RFLAG   sonic vortex flag                            [boolean] 

    
    Outputs:           
    U       X velocity        [unitless]
    V       Y velocity        [unitless]
    W       Z velocity        [unitless]
    RFLAG   sonic vortex flag [boolean] 

    Properties Used:
    N/A
    """      
    
    CPI  = 2 * np.pi
    ARG1 = XSQ1 - RO1
    ARG2 = XSQ2 - RO2
    
    RAD1[ARG1>0.] = np.sqrt(ARG1[ARG1>0.])
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
    RAD1[X1<TOL]    = 0.0
    
    REPS = CUTOFF*XSQ1
    FRAD = RAD1

    bool1[RO1>REPS] = False
    FB1 = (T*X1-B2*Y1)/FRAD
    FT1 = X1/(FRAD*RTV1)
    FT1[RTV1<TOLSQ] = 0.
    
    # Use the boolean to turn things off
    FB1[np.isnan(FB1)] = 1.
    FT1[np.isnan(FT1)] = 1.
    FB1[np.isinf(FB1)] = 1.
    FT1[np.isinf(FT1)] = 1.    
    FB1 = FB1*bool1
    FT1 = FT1*bool1
    
    # Round 2
    # Create a boolean for various conditions for F2 that goes to zero
    bool2 = np.ones_like(B2) * True
    bool2[X2<TOL]   = False
    bool2[RAD2==0.] = False
    RAD2[X2<TOL]    = 0.0
    
    REPS = CUTOFF *XSQ2
    FRAD = RAD2    
    
    bool2[RO2>REPS] = False
    FB2 = (T *X2 - B2 *Y2) /FRAD
    FT2 = X2 /(FRAD *RTV2)
    FT2[RTV2<TOLSQ] = 0.
    
    # Use the boolean to turn things off
    FB2[np.isnan(FB2)] = 1.
    FT2[np.isnan(FT2)] = 1.
    FB2[np.isinf(FB2)] = 1.
    FT2[np.isinf(FT2)] = 1.    
    FB2 = FB2*bool2
    FT2 = FT2*bool2
    
    QB = (FB1 - FB2) /DENOM
    U  = ZETAPI *QB
    V  = ZETAPI *(FT1 - FT2 - QB *T)
    W  = - (QB *XTY + FT1 *Y1 - FT2 *Y2) /CPI    
    
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
    
    # DETERMINE IF TRANSVERSE VORTEX LEG OF HORSESHOE ASSOCIATED TO THE
    # CONTROL POINT UNDER CONSIDERATION IS SONIC (SWEPT PARALLEL TO MACH
    # LINE)? IF SO THEN RFLAG = 0.0, OTHERWISE RFLAG = 1.0.
    size   = n_cp*n_w
    n_mach = int(len(B2)/(size**2))
    T2     = T*T
    T2_1d  = T2[0:size*n_mach]
    T2F    = np.zeros_like(T2_1d)
    T2A    = np.zeros_like(T2_1d)    
    
    F_ind  = np.linspace(0,size*n_mach-1,size*n_mach,dtype=int)
    F_mask = np.ones_like(F_ind,dtype=bool)
    F_mask[(n_cw-1)::n_cw] = False
    F_ind  = F_ind[F_mask]   

    
    A_ind  = np.linspace(0,size*n_mach-1,size*n_mach,dtype=int)
    A_mask = np.ones_like(A_ind,dtype=bool)
    A_mask[::n_cw] = False
    A_ind  = A_ind[A_mask]
    
    T2F[A_ind] = T2_1d[F_ind]
    T2A[F_ind] = T2_1d[A_ind]
    
    # Zero out terms on the LE and TE
    T2F[(n_cw-1)::n_cw] = 0.
    T2A[0::n_cw]        = 0.

    # Create a smaller B2 vector
    B2_inds = np.array(np.ravel(np.atleast_2d(np.linspace(0,size-1,size)).T + (size**2)*np.linspace(0,n_mach-1,n_mach),order='F'),dtype=int)
    B2_1D   = B2[B2_inds]

    TRANS = (B2_1D-T2F)*(B2_1D-T2A)
    FLAG  = np.zeros_like(TRANS)
    FLAG[TRANS<0] = 1
    FLAG_bool = np.array(FLAG,dtype=bool)
    RFLAG[TRANS<0] = 0.

    # COMPUTE THE GENERALIZED PRINCIPAL PART OF THE VORTEX-INDUCED VELOCITY INTEGRAL, WWAVE.
    # FROM LINE 2647 VORLAX, the IR .NE. IRR means that we're looking at vortices that affect themselves
    WWAVE = np.zeros_like(W)
    COX   = CHORD /RNMAX
    WWAVE[B2>T2] = - 0.5 *np.sqrt(B2[B2>T2] -T2[B2>T2] )/COX[B2>T2] 
    
    W = W + EYE*WWAVE    
    
    # IF CONTROL POINT BELONGS TO A SONIC HORSESHOE VORTEX, AND THE
    # SENDING ELEMENT IS SUCH HORSESHOE, THEN MODIFY THE NORMALWASH
    # COEFFICIENTS IN SUCH A WAY THAT THE STRENGTH OF THE SONIC VORTEX
    # WILL BE THE AVERAGE OF THE STRENGTHS OF THE HORSESHOES IMMEDIATELY
    # IN FRONT OF AND BEHIND IT.
    
    # Zero out the row
    FLAG_bool_rep     = np.repeat(FLAG_bool,size)
    W[FLAG_bool_rep]  = 0. # Default to zero

    # The self velocity goes to 2
    FLAG_bool_split   = np.array(np.split(FLAG_bool,n_mach))
    FLAG_ind          = np.array(np.where(FLAG_bool_split))
    squares           = np.tile(np.atleast_3d(np.zeros((size,size))),n_mach)
    squares[FLAG_ind[1],FLAG_ind[1],FLAG_ind[0]] = 1
    squares           = np.ravel(squares,order='F')
    
    FLAG_bool_self    = np.where(squares==1)[0]
    W[FLAG_bool_self] = 2. # It's own value, -2
    
    # The panels before and after go to -1
    FLAG_bool_bef = FLAG_bool_self - 1
    FLAG_bool_aft = FLAG_bool_self + 1
    W[FLAG_bool_bef] = -1.
    W[FLAG_bool_aft] = -1.

    return U, V, W, RFLAG


def supersonic_in_plane(RAD1,RAD2,Y1,Y2,TOL,XTY,CPI):
    """  This computes the induced velocities at each control point 
    in the special case where the vortices lie in the same plane
    
    Assumptions: 
    Trailing vortex legs infinity are alligned to freestream

    Source:  
    1. Miranda, Luis R., Robert D. Elliot, and William M. Baker. "A generalized vortex 
    lattice method for subsonic and supersonic flow applications." (1977). (NASA CR)
    
    2. VORLAX Source Code

    Inputs: 
    RAD1    array of zeros                               [-]
    RAD2    array of zeros                               [-]
    Y1      Y coordinate of the left side of the vortex  [m]
    Y2      Y coordinate of the right side of the vortex [m]
    TOL     coefficient                                  [-]
    XTY     AXIAL DISTANCE BETWEEN PROJECTION OF RECEIVING POINT ONTO HORSESHOE PLANE AND EXTENSION OF SKEWED LEG [m]
    CPI     2 Pi                                         [radians]

    
    Outputs:           
    U       X velocity       [unitless]
    V       Y velocity       [unitless]
    W       Z velocity       [unitless]

    Properties Used:
    N/A
    """    
    
    F1 = np.zeros_like(RAD1)
    F2 = np.zeros_like(RAD2)
    
    F1[np.abs(Y1)>TOL] = RAD1[np.abs(Y1)>TOL]/Y1[np.abs(Y1)>TOL]
    F2[np.abs(Y2)>TOL] = RAD2[np.abs(Y2)>TOL]/Y2[np.abs(Y2)>TOL]
    
    U = np.zeros_like(RAD1)
    V = np.zeros_like(RAD1)
    W = np.zeros_like(RAD1)
    W[np.abs(XTY)>TOL] = (-F1[np.abs(XTY)>TOL] + F2[np.abs(XTY)>TOL])/(XTY[np.abs(XTY)>TOL]*CPI)
    
    return U, V, W