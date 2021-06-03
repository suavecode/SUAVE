## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# VLM.py
# 
# Created:  Oct 2020, E. Botero
# Modified: May 2021, E. Botero     

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
import numpy as np 
from SUAVE.Core import Data
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wing_induced_velocity      import compute_wing_induced_velocity
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_wing_vortex_distribution  import generate_wing_vortex_distribution, compute_panel_area, compute_unit_normal
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_RHS_matrix                 import compute_RHS_matrix 

# ----------------------------------------------------------------------
#  Vortex Lattice
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def VLM(conditions,settings,geometry):
    """Uses the vortex lattice method to compute the lift, induced drag and moment coefficients  

    Assumptions: None

    Source:
    1. Miranda, Luis R., Robert D. Elliot, and William M. Baker. "A generalized vortex 
    lattice method for subsonic and supersonic flow applications." (1977). (NASA CR)
    
    2. VORLAX Source Code

    Inputs:
    geometry.
       reference_area                          [m^2]
       wing.
         spans.projected                       [m]
         chords.root                           [m]
         chords.tip                            [m]
         sweeps.quarter_chord                  [radians]
         taper                                 [Unitless]
         twists.root                           [radians]
         twists.tip                            [radians]
         symmetric                             [Boolean]
         aspect_ratio                          [Unitless]
         areas.reference                       [m^2]
         vertical                              [Boolean]
         origin                                [m]
       fuselage.
        origin                                 [m]
        width                                  [m]
        heights.maximum                        [m]      
        lengths.nose                           [m]    
        lengths.tail                           [m]     
        lengths.total                          [m]     
        lengths.cabin                          [m]     
        fineness.nose                          [Unitless]
        fineness.tail                          [Unitless]
        
       settings.number_spanwise_vortices       [Unitless]
       settings.number_chordwise_vortices      [Unitless]
       settings.use_surrogate                  [Unitless]
       settings.propeller_wake_model           [Unitless]
       conditions.aerodynamics.angle_of_attack [radians]
       conditions.freestream.mach_number       [Unitless]
       
    Outputs:                                   
    CL                                         [Unitless]
    Cl                                         [Unitless]
    CDi                                        [Unitless]
    Cdi                                        [Unitless]
    CM                                         [Unitless]
    CP                                         [Unitless]

    Properties Used:
    N/A
    """ 
    
    # unpack settings
    n_sw       = settings.number_spanwise_vortices    
    n_cw       = settings.number_chordwise_vortices 
    pwm        = settings.propeller_wake_model
    ito        = settings.initial_timestep_offset
    nts        = settings.number_of_wake_timesteps 
    wdt        = settings.wake_development_time   
    K_SPC      = settings.leading_edge_suction_multiplier
    Sref       = geometry.reference_area  

    # define point about which moment coefficient is computed
    if 'main_wing' in geometry.wings:
        c_bar      = geometry.wings['main_wing'].chords.mean_aerodynamic
        x_mac      = geometry.wings['main_wing'].aerodynamic_center[0] + geometry.wings['main_wing'].origin[0][0]
        z_mac      = geometry.wings['main_wing'].aerodynamic_center[2] + geometry.wings['main_wing'].origin[0][2]
        w_span     = geometry.wings['main_wing'].spans.projected
    else:
        c_bar  = 0.
        x_mac  = 0.
        w_span = 0.
        for wing in geometry.wings:
            if wing.vertical == False:
                if c_bar <= wing.chords.mean_aerodynamic:
                    c_bar  = wing.chords.mean_aerodynamic
                    x_mac  = wing.aerodynamic_center[0] + wing.origin[0][0]
                    z_mac  = wing.aerodynamic_center[2] + wing.origin[0][2]
                    w_span = wing.spans.projected

    x_cg       = geometry.mass_properties.center_of_gravity[0][0]
    z_cg       = geometry.mass_properties.center_of_gravity[0][2]
    if x_cg == 0.0:
        x_m = x_mac 
        z_m = z_mac
    else:
        x_m = x_cg
        z_m = z_cg
        
        
    aoa  = conditions.aerodynamics.angle_of_attack   # angle of attack  
    mach = conditions.freestream.mach_number         # mach number
    ones = np.atleast_2d(np.ones_like(mach)) 
    len_mach = len(mach)

    # ---------------------------------------------------------------------------------------
    # STEPS 1-9: Generate Panelization and Vortex Distribution
    # ------------------ --------------------------------------------------------------------    
    # generate vortex distribution (VLM steps 1-9)
    VD   = generate_wing_vortex_distribution(geometry,settings)  

    # Unpack coordinates (VD)
    n_w  = VD.n_w    
    n_cp = VD.n_cp  

    YAH = VD.YAH*1.
    ZAH = VD.ZAH
    ZBH = VD.ZBH    
    YBH = VD.YBH*1.
    
    XA1 = VD.XA1*1.
    XB1 = VD.XB1*1.
    YA1 = VD.YA1
    YB1 = VD.YB1    
    ZA1 = VD.ZA1
    ZB1 = VD.ZB1 
    
    XA2 = VD.XA2
    XB2 = VD.XB2
    YA2 = VD.YA2
    YB2 = VD.YB2    
    ZA2 = VD.ZA2
    ZB2 = VD.ZB2 
    
    XCH = VD.XCH
    YCH = VD.YCH
    ZCH = VD.ZCH
    
    XA_TE =  VD.XA_TE
    XB_TE =  VD.XB_TE
    YA_TE =  VD.YA_TE
    YB_TE =  VD.YB_TE
    ZA_TE =  VD.ZA_TE
    ZB_TE =  VD.ZB_TE     

    # additional VD preprocessing
    # from here on, VD will also be used to hold some processed information about geometry
    # for the easier passage of this information into functions
    XBAR    = np.ones(n_sw*n_w) * x_m
    ZBAR    = np.ones(n_sw*n_w) * z_m 
    
    xa      = np.array(np.atleast_2d(VD.XAH*1.),dtype=np.float32)
    xb      = np.array(np.atleast_2d(VD.XBH*1.),dtype=np.float32)
    xc      = 0.5*(xa+xb)    
    xo      = np.array(np.atleast_2d(VD.XC*1.),dtype=np.float32).T 
    xobar   = (xo - xc)    
    shape   = np.shape(xobar)
    shape_0 = shape[0]    
    RNMAX       = n_cw * 1 # number of chordwise panels
    LE_A_pts_x  = np.array(np.atleast_2d(VD.XA1*1.),dtype=np.float32)[:,0::n_cw]
    LE_B_pts_x  = np.array(np.atleast_2d(VD.XB1*1.),dtype=np.float32)[:,0::n_cw]
    LE_A_pts_z  = np.array(np.atleast_2d(VD.ZA1*1.),dtype=np.float32)[:,0::n_cw]
    LE_B_pts_z  = np.array(np.atleast_2d(VD.ZB1*1.),dtype=np.float32)[:,0::n_cw]    
    LE_X        = (LE_A_pts_x+LE_B_pts_x)/2
    LE_Z        = (LE_A_pts_z+LE_B_pts_z)/2
    TE_X        = (XB_TE + XA_TE)/2
    TE_Z        = (ZB_TE + ZA_TE)/2
    LE_X        = np.repeat(LE_X,n_cw,axis=1)
    LE_Z        = np.repeat(LE_Z,n_cw,axis=1)    
    CHORD       = np.sqrt((TE_X-LE_X)**2 + (TE_Z-LE_Z)**2)
    CHORD       = np.repeat(CHORD,shape_0,axis=0)    
    
    X1c   = (XA1+XB1)/2
    X2c   = (XA2+XB2)/2
    Z1c   = (ZA1+ZB1)/2
    Z2c   = (ZA2+ZB2)/2
    SLOPE = (Z2c - Z1c)/(X2c - X1c)
    SLE   = SLOPE[0::n_cw]
    
    # pack vortex distribution 
    VD.XBAR  = XBAR * 1
    VD.ZBAR  = ZBAR * 1   
    VD.RNMAX = RNMAX
    VD.LE_X  = LE_X
    VD.LE_Z  = LE_Z
    VD.CHORD = CHORD
    VD.SLOPE = SLOPE*1
    VD.SLE   = SLE*1
    geometry.vortex_distribution = VD
    
    # ---------------------------------------------------------------------------------------
    # STEP 10: Generate A and RHS matrices from VD and geometry
    # ------------------ --------------------------------------------------------------------    
    # Compute flow tangency conditions
    phi   = np.arctan((VD.ZBC - VD.ZAC)/(VD.YBC - VD.YAC))*ones # dihedral angle 
    delta = np.arctan((VD.ZC - VD.ZCH)/((VD.XC - VD.XCH)*ones)) # mean camber surface angle 

    # Build the RHS vector 
    rhs = compute_RHS_matrix(n_sw,n_cw,delta,phi,conditions,geometry,pwm,ito,wdt,nts) 
    RHS     = rhs.RHS*1
    ONSET   = rhs.ONSET*1
    
    # Build induced velocity matrix, C_mn
    # This is not affected by AoA, so we can use unique mach numbers only
    m_unique, inv = np.unique(mach,return_inverse=True)
    m_unique      = np.atleast_2d(m_unique).T
    C_mn_small, s, CHORD, RFLAG_small, ZETA, EW_small = compute_wing_induced_velocity(VD,n_sw,n_cw,m_unique)
    
    C_mn  = C_mn_small[inv,:,:,:]
    RFLAG = RFLAG_small[inv,:]
    EW    = EW_small[inv,:,:]

    # Turn off sonic vortices when Mach>1
    RHS = RHS*RFLAG
    
    # Build Aerodynamic Influence Coefficient Matrix
    A =   np.multiply(C_mn[:,:,:,0],np.atleast_3d(np.sin(delta)*np.cos(phi))) \
        + np.multiply(C_mn[:,:,:,1],np.atleast_3d(np.cos(delta)*np.sin(phi))) \
        - np.multiply(C_mn[:,:,:,2],np.atleast_3d(np.cos(phi)*np.cos(delta)))   # validated from book eqn 7.42      

    # Compute vortex strength  
    gamma = np.linalg.solve(A,RHS)

    #rename GAMMA and multiply by -1 to match VORLAX
    GAMMA = np.array(gamma)*(-1)

    # ---------------------------------------------------------------------------------------
    # STEP 11: Compute Pressure Coefficient
    # ------------------ --------------------------------------------------------------------   
    #VORLAX subroutine = PRESS

    #Inputs for sideslip/acceleration
    #For angular values, VORLAX uses degrees by default to radians via DTR (degrees to rads). 
    #SUAVE uses radians and its Units system. All algular variables will be in radians or var*Units.degrees
    PSI       = conditions.aerodynamics.sideslip_angle     
    PITCHQ    = conditions.stability.dynamic.pitch_rate              
    ROLLQ     = conditions.stability.dynamic.roll_rate             
    YAWQ      = conditions.stability.dynamic.yaw_rate 
    VINF      = conditions.freestream.velocity
                  
    # spanwise strip exposure flag, always 0 for SUAVE's infinitely thin airfoils. Needs to change if thick airfoils added
    RJTS = 0                         
    
    # COMPUTE FREE-STREAM AND ONSET FLOW PARAMETERS. Used throughout the remainder of VLM
    B2     = np.tile((mach**2 - 1),n_cp)
    SINALF = np.sin(aoa)
    COSALF = np.cos(aoa)
    SINPSI = np.sin(PSI)
    COPSI  = np.cos(PSI)
    COSIN  = COSALF *SINPSI *2.0
    COSINP = COSALF *SINPSI
    COSCOS = COSALF *COPSI
    PITCH  = PITCHQ /VINF
    ROLL   = ROLLQ /VINF
    YAW    = YAWQ /VINF    
    
    CHORD  = CHORD[0,:]
    CHORD_strip = CHORD[0::n_cw]     

    # Panel Dihedral Angle, using AH and BH location. Similar to COD and SID (later) except for signs
    D   = np.sqrt((YAH-YBH)**2+(ZAH-ZBH)**2)[0::n_cw]
    COS_DL = (YBH-YAH)[0::n_cw]/D
    SIN_DL = (ZBH-ZAH)[0::n_cw]/D

    # COMPUTE EFFECT OF SIDESLIP on DCP intermediate variables. needs change if cosine chorwise spacing added
    FORAXL = COSCOS
    FORLAT = COSIN
    
    TAN_LE = (XB1[0:n_cp*n_w:n_cw] - XA1[0:n_cp*n_w:n_cw])/ \
                np.sqrt((ZB1[0:n_cp*n_w:n_cw]-ZA1[0:n_cp*n_w:n_cw])**2 + \
                        (YB1[0:n_cp*n_w:n_cw]-YA1[0:n_cp*n_w:n_cw])**2)  
    TAN_TE = (XB_TE - XA_TE)/ np.sqrt((ZB_TE-ZA_TE)**2 + (YB_TE-YA_TE)**2) # _TE variables already have np.repeat built in 
    TAN_LE = np.broadcast_to(np.repeat(TAN_LE,n_cw),np.shape(B2)) 
    TAN_TE = np.broadcast_to(TAN_TE,np.shape(B2))    
    
    TNL    = TAN_LE * 1 # VORLAX's SIGN variable not needed, as these are taken directly from geometry
    TNT    = TAN_TE * 1
    
    XIA    = np.broadcast_to(np.tile(np.arange(n_cw)/RNMAX,     n_sw*n_w), np.shape(B2))
    XIB    = np.broadcast_to(np.tile(np.arange(1,n_cw+1)/RNMAX, n_sw*n_w), np.shape(B2))
    TANA   = TNL *(1. - XIA) + TNT *XIA
    TANB   = TNL *(1. - XIB) + TNT *XIB
    
    # cumsum GANT loop if KTOP > 0 (don't actually need KTOP with vectorized arrays and np.roll)
    GFX    = np.tile((1 /CHORD), (len_mach,1))
    GANT   = (GFX*GAMMA).reshape(-1,n_sw, n_cw)
    GANT   = np.cumsum(GANT,axis=2).reshape(len_mach,-1)
    GANT   = np.roll(GANT,1)
    GANT[:,::n_cw]   = 0 
    
    GLAT   = GANT *(TANA - TANB) - GFX *GAMMA *TANB
    cos_DL  = np.broadcast_to(np.repeat(COS_DL,n_cw),np.shape(B2))
    DCPSID = FORLAT * cos_DL *GLAT /(XIB - XIA)
    FACTOR = FORAXL + ONSET
    
    
    # COMPUTE LOAD COEFFICIENT
    GNET = GAMMA*FACTOR
    GNET = GNET *RNMAX /CHORD
    DCP  = 2*GNET + DCPSID
    CP   = DCP

    # ---------------------------------------------------------------------------------------
    # STEP 12: Compute aerodynamic coefficients 
    # ------------------ -------------------------------------------------------------------- 
    #VORLAX subroutine = AERO

    # Work panel by panel
    SURF = np.array(VD.wing_areas)
    SREF = Sref  

    # Flip coordinates on the other side of the wing
    boolean = YBH<0. 
    XA1[boolean], XB1[boolean] = XB1[boolean], XA1[boolean]
    YAH[boolean], YBH[boolean] = YBH[boolean], YAH[boolean]

    # Leading edge sweep and trailing edge sweep. VORLAX does it panel by panel. This will be spanwise.
    TLE = TAN_LE *1
    T2  = TLE*TLE
    STB = np.zeros_like(B2)
    STB[B2<T2] = np.sqrt(T2[B2<T2]-B2[B2<T2])
    STB = STB[:,0::n_cw]    
    
    # DL IS THE DIHEDRAL ANGLE (WITH RESPECT TO THE X-Y PLANE) OF
    # THE IR STREAMWISE STRIP OF HORSESHOE VORTICES. 
    COD = np.cos(phi[0,0::n_cw])  # Just the LE values
    SID = np.sin(phi[0,0::n_cw])  # Just the LE values

    # Now on to each strip
    PION = 2.0 /RNMAX
    ADC  = 0.5*PION

    # XLE = LOCATION OF FIRST VORTEX MIDPOINT IN FRACTION OF CHORD.
    XLE = 0.125 *PION
    
    GAF = 0.5 + 0.5 *RJTS**2

    # CORMED IS LENGTH OF STRIP CENTERLINE BETWEEN LOAD POINT
    # AND TRAILING EDGE? THIS PARAMETER IS USED IN THE COMPUTATION
    # OF THE STRIP ROLLING COUPLE CONTRIBUTION DUE TO SIDESLIP.
    X      = XCH                       #x-coord of load point (horseshoe centroid)
    XTE    = (VD.XA_TE + VD.XB_TE)/2   #Trailing edge x-coord behind the control point?    
    CORMED = XTE - X   

    # SINF REFERENCES THE LOAD CONTRIBUTION OF IRT-VORTEX TO THE
    # STRIP NOMINAL AREA, I.E., AREA OF STRIP ASSUMING CONSTANT
    # (CHORDWISE) HORSESHOE SPAN.    
    SINF = ADC * DCP # The horshoe span lengths have been removed since VST/VSS == 1 always

    # Split into chordwise strengths and sum into strips
    # SICPLE = COUPLE (ABOUT STRIP CENTERLINE) DUE TO SIDESLIP.
    CNC    = np.array(np.split(np.reshape(SINF,(-1,n_cw)).sum(axis=1),len(mach)))
    SICPLE = np.array(np.split(np.reshape(SINF*CORMED,(-1,n_cw)).sum(axis=1),len(mach)))

    # COMPUTE SLOPE (TX) WITH RESPECT TO X-AXIS AT LOAD POINTS BY INTER
    # POLATING BETWEEN CONTROL POINTS AND TAKING INTO ACCOUNT THE LOCAL
    # INCIDENCE.    
    RK   = np.tile(np.linspace(1,n_cw,n_cw),n_sw*n_w)
    XX   = (RK - .75) *PION /2.0
    TX    = SLOPE - ZETA
    CAXL  = -SINF*TX/(1.0+TX**2) # These are the axial forces on each panel
    BMLE  = (XLE-XX)*SINF        # These are moment on each panel

    # Sum onto the panel
    CAXL = np.array(np.split(np.reshape(CAXL,(-1,n_cw)).sum(axis=1),len_mach))
    BMLE = np.array(np.split(np.reshape(BMLE,(-1,n_cw)).sum(axis=1),len_mach))
    
    SICPLE *= (-1) * COSIN * COD * GAF
       
    DCP_LE = DCP[:,0::n_cw]
    
    # COMPUTE LEADING EDGE THRUST COEFF. (CSUC) BY CALCULATING
    # THE TOTAL INDUCED FLOW AT THE LEADING EDGE. THIS COMPUTATION
    # ONLY PERFORMED FOR COSINE CHORDWISE SPACING (LAX = 0).    
    # ** TO DO ** Add cosine spacing (earlier in VLM) to properly capture the magnitude of these effects
    CLE = compute_rotation_effects(settings, EW, GAMMA, len_mach, X, CHORD, XLE, XBAR, 
                                   rhs, COSINP, SINALF, PITCH, ROLL, YAW, STB, RNMAX)    
    
    # Leading edge suction multiplier. See documentation. This is a negative integer if used
    # Default to 1 unless specified otherwise
    SPC  = K_SPC*np.ones_like(DCP_LE)
    
    # If the vehicle is subsonic and there is vortex lift enabled then SPC changes to -1
    VL   = np.repeat(VD.vortex_lift,n_sw)
    m_b  = np.atleast_2d(mach[:,0]<1.)
    SPC_cond = VL*m_b.T
    SPC[SPC_cond] = -1.
    
    CLE  = CLE + 0.5* DCP_LE *np.sqrt(XLE)
    CSUC = 0.5*np.pi*np.abs(SPC)*(CLE**2)*STB

    # TFX AND TFZ ARE THE COMPONENTS OF LEADING EDGE FORCE VECTOR ALONG
    # ALONG THE X AND Z BODY AXES.   
    SLE  = SLOPE[0::n_cw]
    ZETA = ZETA[0::n_cw] 
    XCOS = np.broadcast_to(np.cos(SLE-ZETA),np.shape(DCP_LE))
    XSIN = np.broadcast_to(np.sin(SLE-ZETA),np.shape(DCP_LE))
    TFX  =  1.*XCOS
    TFZ  = -1.*XSIN

    # If a negative number is used for SPC a different correction is used. See VORLAX documentation for Lan reference
    TFX[SPC<0] = XSIN[SPC<0]*np.sign(DCP_LE)[SPC<0]
    TFZ[SPC<0] = np.abs(XCOS)[SPC<0]*np.sign(DCP_LE)[SPC<0]

    CAXL = CAXL - TFX*CSUC

    # Add a dimension into the suction to be chordwise
    T2_LE = T2[:,0::n_cw]
    CNC   = CNC + CSUC*np.sqrt(1+T2_LE)*TFZ

    # FCOS AND FSIN ARE THE COSINE AND SINE OF THE ANGLE BETWEEN
    # THE CHORDLINE OF THE IR-STRIP AND THE X-AXIS    
    FCOS = np.cos(ZETA)
    FSIN = np.sin(ZETA)

    # BFX, BFY, AND BFZ ARE THE COMPONENTS ALONG THE BODY AXES
    # OF THE STRIP FORCE CONTRIBUTION.
    BFX = -  CNC *FSIN + CAXL *FCOS
    BFY = - (CNC *FCOS + CAXL *FSIN) *SID
    BFZ =   (CNC *FCOS + CAXL *FSIN) *COD

    # CONVERT CNC FROM CN INTO CNC (COEFF. *CHORD).
    CNC  = CNC  * CHORD_strip
    BMLE = BMLE * CHORD_strip

    # BMX, BMY, AND BMZ ARE THE COMPONENTS ALONG THE BODY AXES
    # OF THE STRIP MOMENT (ABOUT MOM. REF. POINT) CONTRIBUTION.
    X      = ((VD.XAH+VD.XBH)/2)[0::n_cw]  # These are all LE values
    Y      = ((VD.YAH+VD.YBH)/2)[0::n_cw]  # These are all LE values
    Z      = ((VD.ZAH+VD.ZBH)/2)[0::n_cw]  # These are all LE values
    BMX    = BFZ * Y - BFY * (Z - ZBAR)
    BMX    = BMX + SICPLE
    BMY    = BMLE * COD + BFX * (Z - ZBAR) - BFZ * (X - XBAR)
    BMZ    = BMLE * SID - BFX * Y + BFY * (X - XBAR)
    CDC    = BFZ * SINALF +  (BFX *COPSI + BFY *SINPSI) * COSALF
    CDC    = CDC * CHORD_strip
    CMTC   = BMLE + CNC * (0.25 - XLE) #doesn't affect coefficients, but is in VORLAX

    ES    = 2*s[0,0::n_cw]
    STRIP = ES *CHORD_strip
    LIFT  = (BFZ *COSALF - (BFX *COPSI + BFY *SINPSI) *SINALF)*STRIP
    DRAG  = CDC*ES 
    MOMENT = STRIP * (BMY *COPSI - BMX *SINPSI)  
    FN    = CNC *ES                    #doesn't affect coefficients, but is in VORLAX
    FY    = (BFY *COPSI - BFX *SINPSI) *STRIP
    RM     = STRIP *(BMX *COSALF *COPSI + BMY *COSALF *SINPSI + BMZ *SINALF)
    YM     = STRIP *(BMZ *COSALF - (BMX *COPSI + BMY *SINPSI) *SINALF)
    XSUC   = CSUC *STRIP /SURF         #doesn't affect coefficients, but is in VORLAX

    # Now calculate the coefficients for each wing
    cl_y     = LIFT/CHORD_strip/ES
    cdi_y    = DRAG/CHORD_strip/ES
    CL_wing  = np.array(np.split(np.reshape(LIFT,(-1,n_sw)).sum(axis=1),len(mach)))/SURF
    CDi_wing = np.array(np.split(np.reshape(DRAG,(-1,n_sw)).sum(axis=1),len(mach)))/SURF
    Cl_y     = np.swapaxes(np.array(np.array_split(cl_y,n_w,axis=1)),0,1) 
    Cdi_y    = np.swapaxes(np.array(np.array_split(cdi_y,n_w,axis=1)),0,1)   
    alpha_i = np.arctan(Cdi_y/Cl_y) 
    
    # Now calculate total coefficients
    CL       = np.atleast_2d(np.sum(LIFT,axis=1)/SREF).T          # CLTOT in VORLAX
    CDi      = np.atleast_2d(np.sum(DRAG,axis=1)/SREF).T          # CDTOT in VORLAX
    CM       = np.atleast_2d(np.sum(MOMENT,axis=1)/SREF).T/c_bar  # CMTOT in VORLAX

    CYTOT    = np.atleast_2d(np.sum(FY,axis=1)/SREF).T   # total y force coeff
    CRTOT    = np.atleast_2d(np.sum(RM,axis=1)/SREF).T   # total rolling moment coeff
    CRMTOT   = CRTOT/w_span*(-1)                         # an output in VORLAG.LOG
    CNTOT    = np.atleast_2d(np.sum(YM,axis=1)/SREF).T   # total yawing  moment coeff
    CYMTOT   = CNTOT/w_span*(-1)                         # an output in VORLAG.LOG

    # ---------------------------------------------------------------------------------------
    # STEP 12: Pack outputs
    # ------------------ --------------------------------------------------------------------     
    #VORLAX _TOT outputs
    results = Data()
    results.CL         =  CL         
    results.CDi        =  CDi        
    results.CM         =  CM  
    results.CYTOT      =  CYTOT
    results.CRTOT      =  CRTOT
    results.CRMTOT     =  CRMTOT
    results.CNTOT      =  CNTOT
    results.CYMTOT     =  CYMTOT
    
    #other SUAVE outputs
    results.CL_wing    =  CL_wing   
    results.CDi_wing   =  CDi_wing 
    results.cl_y       =  cl_y     
    results.cdi_y      =  cdi_y     
    results.alpha_i    =  alpha_i  
    results.CP         =  CP
    results.gamma      =  GAMMA
    
    #append inputs
    results.conditions = conditions
    results.settings   = settings
    results.geometry   = geometry
    
    return results


def compute_rotation_effects(settings, EW_small, GAMMA, len_mach, X, CHORD, XLE, XBAR, 
                             rhs, COSINP, SINALF, PITCH, ROLL, YAW, STB, RNMAX):
    spacing     = settings.spanwise_cosine_spacing
    n_cw        = settings.number_chordwise_vortices

    # Normally, VORLAX skips this calculation for linear chordwise spacing (the if statement below). 
    # However, since the trends are correct, albeit underestimated, this calculation is being forced
    # here.
    # **TODO** put this check back in when cosine chordwise spacing is added
    ##if spacing == False: # linear spacing is LAX==1 in VORLAX
    ##    return 0 #CLE not calculated till later for linear spacing
    
    # Computate rotational effects (pitch, roll, yaw rates) on LE suction
    # pick leading edge strip values for EW and reshape GAMMA -> gamma accordingly
    EW    = EW_small[: ,0::n_cw, :]
    n_tot_strips = EW.shape[1]
    gamma = np.array(np.split(np.repeat(GAMMA, n_tot_strips, axis=0), len_mach))
    CLE = (EW*gamma).sum(axis=2)
    
    # Up till EFFINC, some of the following values were computed in compute_RHS_matrix().
    #     EFFINC and ALOC are calculated the exact same way, except for the XGIRO term.
    # LOCATE VORTEX LATTICE CONTROL POINT WITH RESPECT TO THE
    # ROTATION CENTER (XBAR, 0, ZBAR). THE RELATIVE COORDINATES
    # ARE XGIRO, YGIRO, AND ZGIRO. 
    XGIRO = X - CHORD*XLE - np.repeat(XBAR, n_cw)
    YGIRO = rhs.YGIRO
    ZGIRO = rhs.ZGIRO  
    
    # VX, VY, VZ ARE THE FLOW ONSET VELOCITY COMPONENTS AT THE LEADING
    # EDGE (STRIP MIDPOINT). VX, VY, VZ AND THE ROTATION RATES ARE
    # REFERENCED TO THE FREE STREAM VELOCITY.    
    VX = rhs.VX
    VY = (COSINP - YAW  *XGIRO + ROLL *ZGIRO)
    VZ = (SINALF - ROLL *YGIRO + PITCH*XGIRO)

    # CCNTL, SCNTL, SID, and COD were computed in compute_RHS_matrix()
    
    # EFFINC = COMPONENT OF ONSET FLOW ALONG NORMAL TO CAMBERLINE AT
    #          LEADING EDGE.
    EFFINC = VX *rhs.SCNTL + VY *rhs.CCNTL *rhs.SID - VZ *rhs.CCNTL *rhs.COD 
    CLE = CLE - EFFINC[:,0::n_cw] 
    CLE = np.where(STB > 0, CLE /RNMAX /STB, CLE)
    
    return CLE
