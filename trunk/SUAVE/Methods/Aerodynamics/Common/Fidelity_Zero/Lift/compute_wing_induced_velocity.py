## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# compute_wing_induced_velocity.py
# 
# Created:  Dec 2020, E. Botero
# Modified: May 2021, E. Botero  
#           Jun 2021, E. Botero  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
import numpy as np 

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_wing_induced_velocity(VD,mach,compute_EW=False):
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
    mach         = np.array(mach,dtype=np.float32)

    # Control points from the VLM 
    XAH   = np.array(np.atleast_2d(VD.XAH*1.),dtype=np.float32)
    YAH   = np.array(np.atleast_2d(VD.YAH*1.),dtype=np.float32)
    ZAH   = np.array(np.atleast_2d(VD.ZAH*1.),dtype=np.float32)
    XBH   = np.array(np.atleast_2d(VD.XBH*1.),dtype=np.float32)
    YBH   = np.array(np.atleast_2d(VD.YBH*1.),dtype=np.float32)
    ZBH   = np.array(np.atleast_2d(VD.ZBH*1.),dtype=np.float32)
    XA1   = np.array(np.atleast_2d(VD.XA1*1.),dtype=np.float32)
    YA1   = np.array(np.atleast_2d(VD.YA1*1.),dtype=np.float32)
    ZA1   = np.array(np.atleast_2d(VD.ZA1*1.),dtype=np.float32)
    XB1   = np.array(np.atleast_2d(VD.XB1*1.),dtype=np.float32)
    YB1   = np.array(np.atleast_2d(VD.YB1*1.),dtype=np.float32)
    ZB1   = np.array(np.atleast_2d(VD.ZB1*1.),dtype=np.float32)
    XA2   = np.array(np.atleast_2d(VD.XA2*1.),dtype=np.float32)
    YA2   = np.array(np.atleast_2d(VD.YA2*1.),dtype=np.float32)
    ZA2   = np.array(np.atleast_2d(VD.ZA2*1.),dtype=np.float32)
    XB2   = np.array(np.atleast_2d(VD.XB2*1.),dtype=np.float32)
    YB2   = np.array(np.atleast_2d(VD.YB2*1.),dtype=np.float32)
    ZB2   = np.array(np.atleast_2d(VD.ZB2*1.),dtype=np.float32)
    XC    = np.array(np.atleast_2d(VD.XC*1.),dtype=np.float32)
    YC    = np.array(np.atleast_2d(VD.YC*1.),dtype=np.float32)
    ZC    = np.array(np.atleast_2d(VD.ZC*1.),dtype=np.float32)
    XA_TE = np.array(np.atleast_2d(VD.XA_TE*1.),dtype=np.float32)
    XB_TE = np.array(np.atleast_2d(VD.XB_TE*1.),dtype=np.float32)
    
    
    # Panel Dihedral Angle, using AH and BH location
    D      = np.sqrt((YAH-YBH)**2+(ZAH-ZBH)**2)
    COS_DL = (YBH-YAH)/D    
    DL     = np.arccos(COS_DL)
    DL[DL>np.pi/2] = DL[DL>np.pi/2] - np.pi # This flips the dihedral angle for the other side of the wing
    
    # -------------------------------------------------------------------------------------------
    # Compute velocity induced by horseshoe vortex segments on every control point by every panel
    # ------------------------------------------------------------------------------------------- 
    # If YBH is negative, flip A and B, ie negative side of the airplane. Vortex order flips
    boolean = YAH>YBH
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
    theta    = np.arctan2(zb-za,yb-ya)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    
    # rotated axes
    x1bar = (xb - xc)
    y1bar = (yb - yc)*costheta + (zb - zc)*sintheta
    
    xobar = (xo - xc)
    yobar = (yo - yc)*costheta + (zo - zc)*sintheta
    zobar =-(yo - yc)*sintheta + (zo - zc)*costheta
    
    # COMPUTE COORDINATES OF RECEIVING POINT WITH RESPECT TO END POINTS OF SKEWED LEG.
    shape   = np.shape(xobar)
    shape_0 = shape[0]
    shape_1 = shape[1]
    s       = np.abs(y1bar)
    t       = x1bar/y1bar  
    s       = np.repeat(s,shape_0,axis=0)
    t       = np.repeat(t,shape_0,axis=0)
    
    X1 = xobar + t*s # In a planar case XC-XAH
    Y1 = yobar + s   # In a planar case YC-YAH
    X2 = xobar - t*s # In a planar case XC-XBH
    Y2 = yobar - s   # In a planar case YC-YBH
    
    # The cutoff hardcoded into vorlax
    CUTOFF = 0.8
    
    # CALCULATE AXIAL DISTANCE BETWEEN PROJECTION OF RECEIVING POINT ONTO HORSESHOE PLANE AND EXTENSION OF SKEWED LEG.
    XTY = xobar - t*yobar
    
    # The notation in this method is flipped from the paper
    B2 = np.atleast_3d(mach**2-1.)
    
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
    U = np.zeros((n_mach,shape_0,shape_1),dtype=np.float32)
    V = np.zeros((n_mach,shape_0,shape_1),dtype=np.float32)
    W = np.zeros((n_mach,shape_0,shape_1),dtype=np.float32)    
    
    if np.sum(sub)>0:
        # COMPUTATION FOR SUBSONIC HORSESHOE VORTEX
        U[sub], V[sub], W[sub] = subsonic(zobar,XSQ1,RO1_sub,XSQ2,RO2_sub,XTY,t,B2_sub,ZSQ,TOLSQ,X1,Y1,X2,Y2,RTV1,RTV2)   

    
    # COMPUTATION FOR SUPERSONIC HORSESHOE VORTEX. some values computed in a preprocessing section in VLM
    sup         = (B2>=0)[:,0,0]
    B2_sup      = B2[sup,:,:]
    RO1_sup     = B2[sup,:,:]*RTV1
    RO2_sup     = B2[sup,:,:]*RTV2
    RNMAX       = VD.panels_per_strip
    CHORD       = VD.chord_lengths
    CHORD       = np.repeat(CHORD,shape_0,axis=0)
    RFLAG       = np.ones((n_mach,shape_1),dtype=np.int8)
    
    if np.sum(sup)>0:
        U[sup], V[sup], W[sup], RFLAG[sup,:] = supersonic(zobar,XSQ1,RO1_sup,XSQ2,RO2_sup,XTY,t,B2_sup,ZSQ,TOLSQ,TOL,TOLSQ2,\
                                                    X1,Y1,X2,Y2,RTV1,RTV2,CUTOFF,CHORD,RNMAX,n_cp,TE_ind,LE_ind)
         
    
    # Rotate into the vehicle frame and pack into a velocity matrix
    C_mn = np.stack([U, V*costheta - W*sintheta, V*sintheta + W*costheta],axis=-1)
    
    
    if compute_EW == True:
        # Calculate the W velocity in the VORLAX frame for later calcs
        # The angles are Dihedral angle of the current panel - dihedral angle of the influencing panel
        COS1   = np.cos(DL.T - DL)
        SIN1   = np.sin(DL.T - DL) 
        WEIGHT = 1
        
        EW = (W*COS1-V*SIN1)*WEIGHT
    else:
        # Assume that this function is being used outside of VLM, EW is not needed
        EW = np.nan
        

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
    
    CPI  = 4 * np.pi
    RAD1 = np.sqrt(XSQ1 - RO1)
    RAD2 = np.sqrt(XSQ2 - RO2)
    
    TBZ  = (T*T-B2)*ZSQ
    DENOM = XTY * XTY + TBZ
    
    TOLSQ = np.broadcast_to(TOLSQ,np.shape(DENOM))
    
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
    
    CPI    = 2 * np.pi
    T2     = T*T
    ZETAPI = Z/CPI
    shape  = np.shape(RO1)
    RAD1   = np.sqrt(XSQ1 - RO1)
    RAD2   = np.sqrt(XSQ2 - RO2)
    
    RAD1[np.isnan(RAD1)] = 0. 
    RAD2[np.isnan(RAD2)] = 0. 
    
    DENOM             = XTY * XTY + (T2 - B2) *ZSQ # The last part of this is the TBZ term
    SIGN              = np.ones(shape,dtype=np.int8)
    SIGN[DENOM<0]     = -1.
    TOLSQ             = np.broadcast_to(TOLSQ,shape)
    DENOM_COND        = np.abs(DENOM)<TOLSQ
    DENOM[DENOM_COND] = SIGN[DENOM_COND]*TOLSQ[DENOM_COND]
    
    # Create a boolean for various conditions for F1 that goes to zero
    bool1           = np.ones(shape,dtype=np.bool8)
    bool1[:,X1<TOL] = False
    bool1[RAD1==0.] = False
    RAD1[:,X1<TOL]  = 0.0
    
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
    bool2           = np.ones(shape,dtype=np.bool8)
    bool2[:,X2<TOL] = False
    bool2[RAD2==0.] = False
    RAD2[:,X2<TOL]  = 0.0
    
    REPS = CUTOFF *XSQ2
    FRAD = RAD2    
    
    bool2[RO2>REPS] = False
    FB2 = (T *X2 - B2 *Y2)/FRAD
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
    in_plane = np.broadcast_to(ZSQ<TOLSQ2,shape)
    RAD1_in  = RAD1[in_plane]
    RAD2_in  = RAD2[in_plane] 
    Y1_in    = Y1[ZSQ<TOLSQ2]
    Y2_in    = Y2[ZSQ<TOLSQ2]
    XTY_in   = XTY[ZSQ<TOLSQ2]
    TOL_in   = TOL[ZSQ<TOLSQ2]
    
    if np.sum(in_plane)>0:
        W_in = supersonic_in_plane(RAD1_in, RAD2_in, Y1_in, Y2_in, TOL_in, XTY_in, CPI)
    else:
        W_in = []

    U[in_plane] = 0.
    V[in_plane] = 0.
    W[in_plane] = W_in
    
    # DETERMINE IF TRANSVERSE VORTEX LEG OF HORSESHOE ASSOCIATED TO THE
    # CONTROL POINT UNDER CONSIDERATION IS SONIC (SWEPT PARALLEL TO MACH
    # LINE)? IF SO THEN RFLAG = 0.0, OTHERWISE RFLAG = 1.0.
    size   = shape[1]
    n_mach = shape[0]    
    T2S = np.atleast_2d(T2[0,:])*np.ones((n_mach,1))
    T2F = np.zeros((n_mach,size))
    T2A = np.zeros((n_mach,size))
    
    # Setup masks
    F_mask = np.ones((n_mach,size),dtype=np.bool8)
    A_mask = np.ones((n_mach,size),dtype=np.bool8)
    F_mask[:,TE_ind] = False
    A_mask[:,LE_ind] = False
    
    # Apply the mask
    T2F[A_mask] = T2S[F_mask]
    T2A[F_mask] = T2S[A_mask]
    
    # Zero out terms on the LE and TE
    T2F[:,TE_ind] = 0.
    T2A[:,LE_ind]        = 0.

    TRANS = (B2[:,:,0]-T2F)*(B2[:,:,0]-T2A)
    
    RFLAG = np.ones((n_mach,size),dtype=np.int8)
    RFLAG[TRANS<0] = 0.
    
    FLAG_bool          = np.zeros_like(TRANS,dtype=bool)
    FLAG_bool[TRANS<0] = True
    FLAG_bool          = np.reshape(FLAG_bool,(n_mach,size,-1))
    

    # COMPUTE THE GENERALIZED PRINCIPAL PART OF THE VORTEX-INDUCED VELOCITY INTEGRAL, WWAVE.
    # FROM LINE 2647 VORLAX, the IR .NE. IRR means that we're looking at vortices that affect themselves
    WWAVE   = np.zeros(shape,dtype=np.float32)
    COX     = CHORD /RNMAX
    eye     = np.eye(n_cp,dtype=np.int8)
    T2      = np.broadcast_to(T2,shape)*eye
    B2_full = np.broadcast_to(B2,shape)*eye
    COX     = np.broadcast_to(COX,shape)*eye
    WWAVE[B2_full>T2] = - 0.5 *np.sqrt(B2_full[B2_full>T2] -T2[B2_full>T2] )/COX[B2_full>T2] 

    W = W + WWAVE    
    
    # IF CONTROL POINT BELONGS TO A SONIC HORSESHOE VORTEX, AND THE
    # SENDING ELEMENT IS SUCH HORSESHOE, THEN MODIFY THE NORMALWASH
    # COEFFICIENTS IN SUCH A WAY THAT THE STRENGTH OF THE SONIC VORTEX
    # WILL BE THE AVERAGE OF THE STRENGTHS OF THE HORSESHOES IMMEDIATELY
    # IN FRONT OF AND BEHIND IT.
    
    # Zero out the row
    FLAG_bool_rep     = np.broadcast_to(FLAG_bool,shape)
    W[FLAG_bool_rep]  = 0. # Default to zero

    # The self velocity goes to 2
    FLAG_bool_split   = np.array(np.split(FLAG_bool.ravel(),n_mach))
    FLAG_ind          = np.array(np.where(FLAG_bool_split))
    squares           = np.zeros((size,size,n_mach))
    squares[FLAG_ind[1],FLAG_ind[1],FLAG_ind[0]] = 1
    squares           = np.ravel(squares,order='F')
    
    FLAG_bool_self    = np.where(squares==1)[0]
    W                 = W.ravel()
    W[FLAG_bool_self] = 2. # It's own value, -2
    
    # The panels before and after go to -1
    FLAG_bool_bef = FLAG_bool_self - 1
    FLAG_bool_aft = FLAG_bool_self + 1
    W[FLAG_bool_bef] = -1.
    W[FLAG_bool_aft] = -1.
    
    W = np.reshape(W,shape)

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
    
    shape = np.shape(RAD2)
    F1    = np.zeros(shape)
    F2    = np.zeros(shape)
    
    reps  = int(shape[0]/np.size(Y1))
    
    Y1  = np.tile(Y1,reps)
    Y2  = np.tile(Y2,reps)
    TOL = np.tile(TOL,reps)
    XTY = np.tile(XTY,reps)
    
    F1[np.abs(Y1)>TOL] = RAD1[np.abs(Y1)>TOL]/Y1[np.abs(Y1)>TOL]
    F2[np.abs(Y2)>TOL] = RAD2[np.abs(Y2)>TOL]/Y2[np.abs(Y2)>TOL]
    
    W = np.zeros(shape)
    W[np.abs(XTY)>TOL] = (-F1[np.abs(XTY)>TOL] + F2[np.abs(XTY)>TOL])/(XTY[np.abs(XTY)>TOL]*CPI)

    return W
