## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_wing_induced_velocity.py
# 
# Created:  Dec 2020, E. Botero
# Modified: May 2021, E. Botero  
#           Jun 2021, E. Botero  
#           May 2022, E. Botero  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
#import numpy as np 
import jax.numpy as jnp
from jax.numpy import where as w
from jax.numpy import newaxis as na
from jax import lax

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_wing_induced_velocity(VD,mach):
    """ This computes the induced velocities at each control point of the vehicle vortex lattice 

    Assumptions: 
    Trailing vortex legs infinity are alligned to freestream
    
    Outside of a call to the VLM() function itself, EW does not need to be computed, as C_mn 
    provides the same information in the body-frame. 

    Source:  
    1. Miranda, Luis R., Robert D. Elliot, and William M. Baker. "A generalized vortex 
    lattice method for subsonic and supersonic flow applications." (1977). (NASA CR)
    
    2. VORLAX Source Code

    Inputs: 
    VD       - vehicle vortex distribution                    [Unitless] 
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
    LE_ind       = VD.leading_edge_indices
    TE_ind       = VD.trailing_edge_indices
    n_cp         = VD.n_cp
    n_mach       = len(mach)
    mach         = jnp.array(mach,dtype=jnp.float32)

    # Control points from the VLM 
    XAH   = jnp.array(jnp.atleast_2d(VD.XAH*1.),dtype=jnp.float32)
    YAH   = jnp.array(jnp.atleast_2d(VD.YAH*1.),dtype=jnp.float32)
    ZAH   = jnp.array(jnp.atleast_2d(VD.ZAH*1.),dtype=jnp.float32)
    XBH   = jnp.array(jnp.atleast_2d(VD.XBH*1.),dtype=jnp.float32)
    YBH   = jnp.array(jnp.atleast_2d(VD.YBH*1.),dtype=jnp.float32)
    ZBH   = jnp.array(jnp.atleast_2d(VD.ZBH*1.),dtype=jnp.float32)
    XA1   = jnp.array(jnp.atleast_2d(VD.XA1*1.),dtype=jnp.float32)
    YA1   = jnp.array(jnp.atleast_2d(VD.YA1*1.),dtype=jnp.float32)
    ZA1   = jnp.array(jnp.atleast_2d(VD.ZA1*1.),dtype=jnp.float32)
    XB1   = jnp.array(jnp.atleast_2d(VD.XB1*1.),dtype=jnp.float32)
    YB1   = jnp.array(jnp.atleast_2d(VD.YB1*1.),dtype=jnp.float32)
    ZB1   = jnp.array(jnp.atleast_2d(VD.ZB1*1.),dtype=jnp.float32)
    XA2   = jnp.array(jnp.atleast_2d(VD.XA2*1.),dtype=jnp.float32)
    YA2   = jnp.array(jnp.atleast_2d(VD.YA2*1.),dtype=jnp.float32)
    ZA2   = jnp.array(jnp.atleast_2d(VD.ZA2*1.),dtype=jnp.float32)
    XB2   = jnp.array(jnp.atleast_2d(VD.XB2*1.),dtype=jnp.float32)
    YB2   = jnp.array(jnp.atleast_2d(VD.YB2*1.),dtype=jnp.float32)
    ZB2   = jnp.array(jnp.atleast_2d(VD.ZB2*1.),dtype=jnp.float32)
    XC    = jnp.array(jnp.atleast_2d(VD.XC*1.),dtype=jnp.float32)
    YC    = jnp.array(jnp.atleast_2d(VD.YC*1.),dtype=jnp.float32)
    ZC    = jnp.array(jnp.atleast_2d(VD.ZC*1.),dtype=jnp.float32)
    XA_TE = jnp.array(jnp.atleast_2d(VD.XA_TE*1.),dtype=jnp.float32)
    XB_TE = jnp.array(jnp.atleast_2d(VD.XB_TE*1.),dtype=jnp.float32)
    
    
    # Panel Dihedral Angle, using AH and BH location
    D      = jnp.sqrt((YAH-YBH)**2+(ZAH-ZBH)**2)
    COS_DL = (YBH-YAH)/D    
    DL     = jnp.arccos(COS_DL)
    DL     = w(DL>(jnp.pi/2),DL - jnp.pi,DL) # This flips the dihedral angle for the other side of the wing
    
    # -------------------------------------------------------------------------------------------
    # Compute velocity induced by horseshoe vortex segments on every control point by every panel
    # ------------------------------------------------------------------------------------------- 
    # If YBH is negative, flip A and B, ie negative side of the airplane. Vortex order flips
    b = YAH>YBH
    XA1, XB1= w(b,XB1,XA1), w(b,XA1,XB1)
    YA1, YB1= w(b,YB1,YA1), w(b,YA1,YB1)
    ZA1, ZB1= w(b,ZB1,ZA1), w(b,ZA1,ZB1)
    XA2, XB2= w(b,XB2,XA2), w(b,XA2,XB2)
    YA2, YB2= w(b,YB2,YA2), w(b,YA2,YB2)
    ZA2, ZB2= w(b,ZB2,ZA2), w(b,ZA2,ZB2)
    XAH, XBH= w(b,XBH,XAH), w(b,XAH,XBH)
    YAH, YBH= w(b,YBH,YAH), w(b,YAH,YBH)
    ZAH, ZBH= w(b,ZBH,ZAH), w(b,ZAH,ZBH)
    
    XA_TE, XB_TE= w(b,XB_TE,XA_TE), w(b,XA_TE,XB_TE)
    
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
    xo = XC.T
    yo = YC.T
    zo = ZC.T
    
    # Incline the vortex
    theta    = jnp.arctan2(zb-za,yb-ya)
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)
    
    # rotated axes
    x1bar = (xb - xc)
    y1bar = (yb - yc)*costheta + (zb - zc)*sintheta
    
    xobar = (xo - xc)
    yobar = (yo - yc)*costheta + (zo - zc)*sintheta
    zobar =-(yo - yc)*sintheta + (zo - zc)*costheta
    
    # COMPUTE COORDINATES OF RECEIVING POINT WITH RESPECT TO END POINTS OF SKEWED LEG.
    shape   = jnp.shape(xobar)
    shape_0 = shape[0]
    shape_1 = shape[1]
    s       = jnp.abs(y1bar)
    t       = x1bar/y1bar  
    s       = jnp.repeat(s,shape_0,axis=0)
    t       = jnp.repeat(t,shape_0,axis=0)
    
    X1 = xobar + t*s # In a planar case XC-XAH
    Y1 = yobar + s   # In a planar case YC-YAH
    X2 = xobar - t*s # In a planar case XC-XBH
    Y2 = yobar - s   # In a planar case YC-YBH
    
    # The cutoff hardcoded into vorlax
    CUTOFF = 0.8
    
    # CALCULATE AXIAL DISTANCE BETWEEN PROJECTION OF RECEIVING POINT ONTO HORSESHOE PLANE AND EXTENSION OF SKEWED LEG.
    XTY = xobar - t*yobar
    
    # The notation in this method is flipped from the paper
    B2 = jnp.atleast_3d(mach**2-1.)
    
    # SET VALUES OF NUMERICAL TOLERANCE CONSTANTS.
    TOL    = s /500.0
    TOLSQ  = TOL *TOL
    TOLSQ2 = 2500.0 *TOLSQ
    ZSQ    = zobar *zobar
    YSQ1   = Y1 *Y1
    YSQ2   = Y2 *Y2
    RTV1   = YSQ1 + ZSQ
    RTV2   = YSQ2 + ZSQ
    XSQ1   = X1 *X1
    XSQ2   = X2 *X2
    
    # Split the vectors into subsonic and supersonic
    sub      = (B2<0)[:,0,0]
    B2_sub   = B2[sub,:,:]
    RO1_sub  = B2_sub*RTV1
    RO2_sub  = B2_sub*RTV2
    
    # ZERO-OUT PERTURBATION VELOCITY COMPONENTS
    U = jnp.zeros((n_mach,shape_0,shape_1),dtype=jnp.float32)
    V = jnp.zeros((n_mach,shape_0,shape_1),dtype=jnp.float32)
    W = jnp.zeros((n_mach,shape_0,shape_1),dtype=jnp.float32)    
    
    if jnp.sum(sub)>0:
        # COMPUTATION FOR SUBSONIC HORSESHOE VORTEX
        U_sub, V_sub, W_sub = subsonic(zobar,XSQ1,RO1_sub,XSQ2,RO2_sub,XTY,t,B2_sub,ZSQ,TOLSQ,X1,Y1,X2,Y2,RTV1,RTV2)
        U = U.at[sub.nonzero()].set(U_sub[sub.nonzero()])
        V = V.at[sub.nonzero()].set(V_sub[sub.nonzero()])
        W = W.at[sub.nonzero()].set(W_sub[sub.nonzero()])
    
    # COMPUTATION FOR SUPERSONIC HORSESHOE VORTEX. some values computed in a preprocessing section in VLM
    sup         = (B2>=0)[:,0,0]
    B2_sup      = B2[sup,:,:]
    RO1_sup     = B2[sup,:,:]*RTV1
    RO2_sup     = B2[sup,:,:]*RTV2
    RNMAX       = VD.panels_per_strip
    CHORD       = VD.chord_lengths
    CHORD       = jnp.repeat(CHORD,shape_0,axis=0)
    RFLAG       = jnp.ones((n_mach,shape_1),dtype=jnp.int8)
    
    if jnp.sum(sup)>0:
        U_sup, V_sup, W_sup, RFLAG_sup = supersonic(zobar,XSQ1,RO1_sup,XSQ2,RO2_sup,XTY,t,B2_sup,ZSQ,TOLSQ,TOL,TOLSQ2,\
                                                    X1,Y1,X2,Y2,RTV1,RTV2,CUTOFF,CHORD,RNMAX,n_cp,TE_ind,LE_ind)
        U = U.at[sup.nonzero()].set(U_sup[sup.nonzero()])
        V = V.at[sup.nonzero()].set(V_sup[sup.nonzero()])
        W = W.at[sup.nonzero()].set(W_sup[sup.nonzero()])
        RFLAG = RFLAG.at[sup.nonzero()].set(1)

    # Rotate into the vehicle frame and pack into a velocity matrix
    C_mn = jnp.stack([U, V*costheta - W*sintheta, V*sintheta + W*costheta],axis=-1)
    
    # Calculate the W velocity in the VORLAX frame for later calcs
    # The angles are Dihedral angle of the current panel - dihedral angle of the influencing panel
    COS1   = jnp.cos(DL.T - DL)
    SIN1   = jnp.sin(DL.T - DL) 
    WEIGHT = 1
    limit_W = W[:,:n_cp,:n_cp] # This limiter is necessary to maintain array shapes.
    limit_V = V[:,:n_cp,:n_cp] # Overflowed shapes would only occur in something like nonuniform prop inflow and it's not used
    
    EW     = (limit_W*COS1-limit_V*SIN1)*WEIGHT    

    return C_mn, s, RFLAG, EW
    
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
    
    CPI  = 4 * jnp.pi
    RAD1 = jnp.sqrt(XSQ1 - RO1)
    RAD2 = jnp.sqrt(XSQ2 - RO2)
    
    TBZ  = (T*T-B2)*ZSQ
    DENOM = XTY * XTY + TBZ
    
    TOLSQ = jnp.broadcast_to(TOLSQ,jnp.shape(DENOM))
    
    DENOM = w(DENOM<TOLSQ,TOLSQ,DENOM)
    
    FB1 = (T *X1 - B2 *Y1) /RAD1
    FT1 = (X1 + RAD1) /(RAD1 *RTV1)
    FT1 = w(RTV1<TOLSQ,0.,FT1)
    
    FB2 = (T *X2 - B2 *Y2) /RAD2
    FT2 = (X2 + RAD2) /(RAD2 *RTV2)
    FT2 = w(RTV2<TOLSQ,0.,FT2)
    
    QB     = (FB1 - FB2) /DENOM
    ZETAPI = Z /CPI
    U      = ZETAPI *QB
    U      = w(ZSQ<TOLSQ,0.,U)
    V      = ZETAPI * (FT1 - FT2 - QB *T)
    V      = w(ZSQ<TOLSQ,0.,V)
    W      = - (QB *XTY + FT1 *Y1 - FT2 *Y2) /CPI
    
    return U, V, W

def supersonic(Z,XSQ1,RO1,XSQ2,RO2,XTY,T,B2,ZSQ,TOLSQ,TOL,TOLSQ2,X1,Y1,X2,Y2,RTV1,RTV2,CUTOFF,CHORD,RNMAX,n_cp,TE_ind, LE_ind):
    """  This computes the induced velocities at each control point 
    of the vehicle vortex lattice for supersonic mach numbers

    Assumptions: 
    Trailing vortex legs infinity are alligned to freestream

    Source:  
    1. Miranda, Luis R., Robert D. Elliot, and William M. Baker. "A generalized vortex 
    lattice method for subsonic and supersonic flow applications." (1977). (NASA CR)
    
    2. VORLAX Source Code

    Inputs: 
    Z            Z relative location of the vortices          [m]
    XSQ1         X1 squared                                   [m^2]
    RO1          coefficient                                  [-]
    XSQ2         X2 squared                                   [m^2]
    RO2          coefficient                                  [-]
    XTY          AXIAL DISTANCE BETWEEN PROJECTION OF RECEIVING POINT ONTO HORSESHOE PLANE AND EXTENSION OF SKEWED LEG [m]
    T            tangent of the horshoe vortex                [-] 
    B2           mach^2-1 (-beta2)                            [-] 
    ZSQ          Z squared                                    [m^2] 
    TOLSQ        coefficient                                  [-]
    X1           X coordinate of the left side of the vortex  [m]
    Y1           Y coordinate of the left side of the vortex  [m]
    X2           X coordinate of the right side of the vortex [m]
    Y2           Y coordinate of the right side of the vortex [m]
    RTV1         coefficient                                  [-]
    RTV2         coefficient                                  [-]
    CUTOFF       coefficient                                  [-]
    CHORD        chord length for a panel                     [m] 
    RNMAX        number of chordwise panels                   [-]
    n_cp         number of control points                     [-]
    TE_ind       indices of the trailing edge                 [-]
    LE_ind       indices of the leading edge                  [-]
    

    
    Outputs:           
    U       X velocity        [unitless]
    V       Y velocity        [unitless]
    W       Z velocity        [unitless]
    RFLAG   sonic vortex flag [boolean] 

    Properties Used:
    N/A
    """      
    
    CPI    = 2 * jnp.pi
    T2     = T*T
    ZETAPI = Z/CPI
    shape  = jnp.shape(RO1)
    RAD1   = jnp.sqrt(XSQ1 - RO1)
    RAD2   = jnp.sqrt(XSQ2 - RO2)
    RAD1   = w(jnp.isnan(RAD1),0.,RAD1) 
    RAD2   = w(jnp.isnan(RAD2),0.,RAD2) 
    
    DENOM           = XTY * XTY + (T2 - B2) *ZSQ # The last part of this is the TBZ term
    SIGN            = jnp.ones(shape,dtype=jnp.int8)
    SIGN            = w(DENOM<0,-1.,SIGN)
    TOLSQ           = jnp.broadcast_to(TOLSQ,shape)
    DENOM_COND      = jnp.abs(DENOM)<TOLSQ
    DENOM           = w(DENOM_COND,SIGN*TOLSQ,DENOM)
    
    # Create a boolean for various conditions for F1 that goes to zero
    bool1           = jnp.ones(shape,dtype=jnp.int8)
    bool1           = w((X1<TOL)[na,:], 0,bool1)
    bool1           = w(RAD1==0.,0,bool1)
    RAD1            = w((X1<TOL)[na,:], 0.,RAD1)
    
    REPS = CUTOFF*XSQ1
    FRAD = RAD1

    bool1 = w(RO1>REPS,0,bool1)
    FB1   = (T*X1-B2*Y1)/FRAD
    FT1   = X1/(FRAD*RTV1)
    FT1   = w(RTV1<TOLSQ,0.,FT1)
    
    # Use the boolean to turn things off
    FB1 = w(jnp.isnan(FB1),1.,FB1)
    FT1 = w(jnp.isnan(FT1),1.,FT1)
    FB1 = w(jnp.isinf(FB1),1.,FB1)
    FT1 = w(jnp.isinf(FT1),1.,FT1)
    FB1 = FB1*bool1
    FT1 = FT1*bool1
    
    # Round 2
    # Create a boolean for various conditions for F2 that goes to zero
    bool2           = jnp.ones(shape,dtype=jnp.int8)
    bool2           = w((X2<TOL)[na,:], 0,bool2)
    bool2           = w(RAD2==0.,0,bool2)
    RAD2            = w((X2<TOL)[na,:], 0.,RAD2)
    
    REPS = CUTOFF *XSQ2
    FRAD = RAD2    
    
    bool2 = w(RO2>REPS, 0, bool2)
    FB2   = (T *X2 - B2 *Y2)/FRAD
    FT2   = X2 /(FRAD *RTV2)
    FT2   = w(RTV2<TOLSQ,0.,FT2)
    
    # Use the boolean to turn things off
    FB2 = w(jnp.isnan(FB2),1.,FB2)
    FT2 = w(jnp.isnan(FT2),1.,FT2)
    FB2 = w(jnp.isinf(FB2),1.,FB2)
    FT2 = w(jnp.isinf(FT2),1.,FT2)
    FB2 = FB2*bool2
    FT2 = FT2*bool2
    
    QB = (FB1 - FB2) /DENOM
    U  = ZETAPI *QB
    V  = ZETAPI *(FT1 - FT2 - QB *T)
    W  = - (QB *XTY + FT1 *Y1 - FT2 *Y2) /CPI    
    
    # COMPUTATION FOR SUPERSONIC HORSESHOE VORTEX WHEN RECEIVING POINT IS IN THE PLANE OF THE HORSESHOE
    in_plane = jnp.broadcast_to(ZSQ<TOLSQ2,shape)
    RAD1_in  = RAD1[in_plane]
    RAD2_in  = RAD2[in_plane] 
    Y1_in    = Y1[ZSQ<TOLSQ2]
    Y2_in    = Y2[ZSQ<TOLSQ2]
    XTY_in   = XTY[ZSQ<TOLSQ2]
    TOL_in   = TOL[ZSQ<TOLSQ2]
    
    if jnp.sum(in_plane)>0:
        W_in = supersonic_in_plane(RAD1_in, RAD2_in, Y1_in, Y2_in, TOL_in, XTY_in, CPI)
    else:
        W_in = []

    U = U.at[in_plane].set(0)
    V = V.at[in_plane].set(0)
    W = W.at[in_plane].set(W_in)

    # DETERMINE IF TRANSVERSE VORTEX LEG OF HORSESHOE ASSOCIATED TO THE
    # CONTROL POINT UNDER CONSIDERATION IS SONIC (SWEPT PARALLEL TO MACH
    # LINE)? IF SO THEN RFLAG = 0.0, OTHERWISE RFLAG = 1.0.
    size   = shape[1]
    n_mach = shape[0]    
    T2S    = jnp.atleast_2d(T2[0,:])*jnp.ones((n_mach,1))
    T2F    = jnp.zeros((n_mach,size))
    T2A    = jnp.zeros((n_mach,size))
    
    # Setup masks
    F_mask = jnp.ones((n_mach,size),dtype=bool)
    A_mask = jnp.ones((n_mach,size),dtype=bool)
    F_mask = F_mask.at[:,TE_ind].set(0)
    A_mask = A_mask.at[:,LE_ind].set(0)
    
    # Apply the mask
    T2F = T2F.at[A_mask].set(T2S[F_mask])
    T2A = T2A.at[F_mask].set(T2S[A_mask])
    
    # Zero out terms on the LE and TE
    T2F = T2F.at[:, TE_ind].set(0)
    T2A = T2A.at[:, LE_ind].set(0)

    # T2F = w(TE_ind[:,na],0.,T2F)
    # T2A = w(LE_ind[:,na],0.,T2A)

    TRANS = (B2[:,:,0]-T2F)*(B2[:,:,0]-T2A)
    
    RFLAG = jnp.ones((n_mach,size),dtype=jnp.int8)
    RFLAG = w(TRANS<0,0.,RFLAG)
    
    FLAG_bool = jnp.zeros_like(TRANS,dtype=bool)
    FLAG_bool = w(TRANS<0,True,FLAG_bool)
    FLAG_bool = jnp.reshape(FLAG_bool,(n_mach,size,-1))
    

    # COMPUTE THE GENERALIZED PRINCIPAL PART OF THE VORTEX-INDUCED VELOCITY INTEGRAL, WWAVE.
    # FROM LINE 2647 VORLAX, the IR .NE. IRR means that we're looking at vortices that affect themselves
    WWAVE   = jnp.zeros(shape,dtype=jnp.float32)
    COX     = CHORD /RNMAX
    eye     = jnp.eye(n_cp,dtype=jnp.int8)
    T2      = jnp.broadcast_to(T2,shape)*eye
    B2_full = jnp.broadcast_to(B2,shape)*eye
    COX     = jnp.broadcast_to(COX,shape)*eye
    WWAVE   = w(B2_full>T2,- 0.5 *jnp.sqrt(B2_full -T2 )/COX,WWAVE)

    W = W + WWAVE    
    
    # IF CONTROL POINT BELONGS TO A SONIC HORSESHOE VORTEX, AND THE
    # SENDING ELEMENT IS SUCH HORSESHOE, THEN MODIFY THE NORMALWASH
    # COEFFICIENTS IN SUCH A WAY THAT THE STRENGTH OF THE SONIC VORTEX
    # WILL BE THE AVERAGE OF THE STRENGTHS OF THE HORSESHOES IMMEDIATELY
    # IN FRONT OF AND BEHIND IT.
    
    # Zero out the row
    FLAG_bool_rep     = jnp.broadcast_to(FLAG_bool,shape)
    W                 = w(FLAG_bool_rep,0.,W) # Default to zero

    # The self velocity goes to 2
    FLAG_bool_split   = jnp.array(jnp.split(FLAG_bool.ravel(),n_mach))
    FLAG_ind          = jnp.array(jnp.where(FLAG_bool_split))
    squares           = jnp.zeros((size,size,n_mach))
    squares           = squares.at[FLAG_ind[1],FLAG_ind[1],FLAG_ind[0]].set(1)
    squares           = jnp.ravel(squares,order='F')
    
    FLAG_bool_self    = jnp.where(squares==1)[0]
    W                 = W.ravel()
    W                 = W.at[FLAG_bool_self].set(2.) # It's own value, -2
    
    # The panels before and after go to -1
    FLAG_bool_bef = FLAG_bool_self - 1
    FLAG_bool_aft = FLAG_bool_self + 1
    W             = W.at[FLAG_bool_bef].set(-1.)
    W             = W.at[FLAG_bool_aft].set(-1.)
    
    W = jnp.reshape(W,shape)

    return U, V, W, RFLAG


def supersonic_in_plane(RAD1,RAD2,Y1,Y2,TOL,XTY,CPI):
    """  This computes the induced velocities at each control point 
    in the special case where the vortices lie in the same plane
    
    Assumptions: 
    Trailing vortex legs infinity are alligned to freestream
    In plane vortices only produce W velocity

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
    W       Z velocity       [unitless]

    Properties Used:
    N/A
    """    
    
    shape = jnp.shape(RAD2)
    F1    = jnp.zeros(shape)
    F2    = jnp.zeros(shape)
    
    reps  = int(shape[0]/jnp.size(Y1))
    
    Y1  = jnp.tile(Y1,reps)
    Y2  = jnp.tile(Y2,reps)
    TOL = jnp.tile(TOL,reps)
    XTY = jnp.tile(XTY,reps)
    
    F1  = w(jnp.abs(Y1)>TOL,RAD1/Y1,F1)
    F2  = w(jnp.abs(Y2)>TOL,RAD2/Y2,F2)
    
    W = jnp.zeros(shape)
    W = w(jnp.abs(XTY)>TOL,(-F1+ F2)/(XTY*CPI),W)

    return W