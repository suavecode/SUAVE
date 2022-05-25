## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# VLM.py
# 
# Created:  Oct 2020, E. Botero
# Modified: May 2021, E. Botero   
#           Jul 2021, A. Blaufox    
#           May 2022, E. Botero  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
import numpy as np
import jax.numpy as jnp
from jax.numpy import where as w
from jax.numpy import newaxis as na
from SUAVE.Core import Data
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wing_induced_velocity      import compute_wing_induced_velocity
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_RHS_matrix                 import compute_RHS_matrix 

# ----------------------------------------------------------------------
#  Vortex Lattice
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def VLM(conditions,settings,geometry):
    """Uses the vortex lattice method to compute the lift, induced drag and moment coefficients.
    
    The user has the option to discretize control surfaces using the boolean settings.discretize_control_surfaces.
    The user should be forwarned that this will cause very slight differences in results for 0 deflection due to
    the slightly different discretization.
    
    The user has the option to use the boundary conditions and induced velocities from either SUAVE
    or VORLAX. See build_RHS in compute_RHS_matrix.py for more details.
    
    By default in Vortex_Lattice, VLM performs calculations based on panel coordinates with float32 precision. 
    The user may also choose to use float16 or float64, but be warned that the latter can be memory intensive.
    
    The user should note that fully capitalized variables correspond to a VORLAX variable of the same name
    
    
    Assumptions:
    The user provides either global discretezation (number_spanwise/chordwise_vortices) or
    separate discretization (wing/fuselage_spanwise/chordwise_vortices) in settings, not both.
    The set of settings not being used should be set to None.
    
    The VLM requires that the user provide a non-zero velocity that matches mach number. For
    surrogate training cases at mach 0, VLM uses a velocity of 1e-6 m/s

    
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
        
    settings.number_spanwise_vortices          [Unitless]  <---|
    settings.number_chordwise_vortices         [Unitless]  <---|
                                                               |--Either/or; see generate_vortex_distribution() for more details
    settings.wing_spanwise_vortices            [Unitless]  <---|
    settings.wing_chordwise_vortices           [Unitless]  <---|
    settings.fuselage_spanwise_vortices        [Unitless]  <---|
    settings.fuselage_chordwise_vortices       [Unitless]  <---|  
       
    settings.use_surrogate                     [Unitless]
    settings.propeller_wake_model              [Unitless]
    settings.discretize_control_surfaces       [Boolean], set to True to generate control surface panels
    settings.use_VORLAX_matrix_calculation     [boolean]
    settings.floating_point_precision          [np.float16/32/64]
       
    conditions.aerodynamics.angle_of_attack    [radians]
    conditions.aerodynamics.side_slip_angle    [radians]
    conditions.freestream.mach_number          [Unitless]
    conditions.freestream.velocity             [m/s]
    conditions.stability.dynamic.pitch_rate    [radians/s]
    conditions.stability.dynamic.roll_rate     [radians/s]
    conditions.stability.dynamic.yaw_rate      [radians/s]
       
    
    Outputs:    
    results.
        CL                                     [Unitless], CLTOT in VORLAX
        CDi                                    [Unitless], CDTOT in VORLAX
        CM                                     [Unitless], CMTOT in VORLAX
        CYTOT                                  [Unitless], Total y force coeff
        CRTOT                                  [Unitless], Rolling moment coeff (unscaled)
        CRMTOT                                 [Unitless], Rolling moment coeff (scaled by w_span)
        CNTOT                                  [Unitless], Yawing  moment coeff (unscaled)
        CYMTOT                                 [Unitless], Yawing  moment coeff (scaled by w_span)
        CL_wing                                [Unitless], CL  of each wing
        CDi_wing                               [Unitless], CDi of each wing
        cl_y                                   [Unitless], CL  of each strip
        cdi_y                                  [Unitless], CDi of each strip
        alpha_i                                [radians] , Induced angle of each strip in each wing (array of numpy arrays)
        CP                                     [Unitless], Pressure coefficient of each panel
        gamma                                  [Unitless], Vortex strengths of each panel

    
    Properties Used:
    N/A
    """ 
    # unpack settings----------------------------------------------------------------
    pwm        = settings.propeller_wake_model
    K_SPC      = settings.leading_edge_suction_multiplier
    Sref       = geometry.reference_area              

    # unpack geometry----------------------------------------------------------------
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
        
    # unpack conditions--------------------------------------------------------------
    aoa       = conditions.aerodynamics.angle_of_attack   # angle of attack  
    mach      = conditions.freestream.mach_number         # mach number
    ones      = jnp.atleast_2d(jnp.ones_like(mach)) 
    len_mach  = len(mach)
    
    #For angular values, VORLAX uses degrees by default to radians via DTR (degrees to rads). 
    #SUAVE uses radians and its Units system. All algular variables will be in radians or var*Units.degrees
    PSI       = conditions.aerodynamics.side_slip_angle     
    PITCHQ    = conditions.stability.dynamic.pitch_rate              
    ROLLQ     = conditions.stability.dynamic.roll_rate             
    YAWQ      = conditions.stability.dynamic.yaw_rate 
    VINF      = conditions.freestream.velocity    
       
    #freestream 0 velocity safeguard
    if not conditions.freestream.velocity.all():
        if settings.use_surrogate:
            velocity                       = conditions.freestream.velocity
            velocity[velocity==0]          = np.ones(len(velocity[velocity==0])) * 1e-6
            conditions.freestream.velocity = velocity
        else:
            raise AssertionError("VLM requires that conditions.freestream.velocity be specified and non-zero")    

    # ---------------------------------------------------------------------------------------
    # STEPS 1-9: Generate Panelization and Vortex Distribution
    # ------------------ --------------------------------------------------------------------    
    
    
    # Unpack vortex distribution
    VD           = geometry.VD
    n_cp         = VD.n_cp 
    n_sw         = VD.n_sw
    CHORD        = VD.chord_lengths
    chord_breaks = VD.chordwise_breaks
    span_breaks  = VD.spanwise_breaks
    RNMAX        = VD.panels_per_strip    
    LE_ind       = VD.leading_edge_indices
    ZETA         = VD.tangent_incidence_angle
    RK           = VD.chordwise_panel_number
    
    exposed_leading_edge_flag = VD.exposed_leading_edge_flag
    
    YAH = VD.YAH*1.  
    YBH = VD.YBH*1.
    
    XA1 = VD.XA1*1.
    XB1 = VD.XB1*1.
    YA1 = VD.YA1
    YB1 = VD.YB1    
    ZA1 = VD.ZA1
    ZB1 = VD.ZB1  
    
    XCH = VD.XCH
    
    XA_TE =  VD.XA_TE
    XB_TE =  VD.XB_TE
    YA_TE =  VD.YA_TE
    YB_TE =  VD.YB_TE
    ZA_TE =  VD.ZA_TE
    ZB_TE =  VD.ZB_TE     
     
    SLOPE = VD.SLOPE
    SLE   = VD.SLE
    D     = VD.D
    
    # Compute X and Z BAR ouside of generate_vortex_distribution to avoid requiring x_m and z_m as inputs
    XBAR    = jnp.ones(sum(LE_ind)) * x_m
    ZBAR    = jnp.ones(sum(LE_ind)) * z_m
    VD.XBAR = XBAR
    VD.ZBAR = ZBAR
    
    # ---------------------------------------------------------------------------------------
    # STEP 10: Generate A and RHS matrices from VD and geometry
    # ------------------ --------------------------------------------------------------------    
    # Compute flow tangency conditions
    phi   = jnp.arctan((VD.ZBC - VD.ZAC)/(VD.YBC - VD.YAC))*ones # dihedral angle 
    delta = jnp.arctan((VD.ZC - VD.ZCH)/((VD.XC - VD.XCH)*ones)) # mean camber surface angle 

    # Build the RHS vector    
    rhs = compute_RHS_matrix(delta,phi,conditions,settings,geometry,pwm) 
    RHS     = rhs.RHS*1
    ONSET   = rhs.ONSET*1

    # Build induced velocity matrix, C_mn
    # This is not affected by AoA, so we can use unique mach numbers only
    m_unique, inv = jnp.unique(mach,return_inverse=True)
    m_unique      = jnp.atleast_2d(m_unique).T
    C_mn_small, s, RFLAG_small, EW_small = compute_wing_induced_velocity(VD,m_unique,compute_EW=True)
    
    C_mn  = C_mn_small[inv,:,:,:]
    RFLAG = RFLAG_small[inv,:]
    EW    = EW_small[inv,:,:]

    # Turn off sonic vortices when Mach>1
    RHS = RHS*RFLAG
    
    # Build Aerodynamic Influence Coefficient Matrix
    use_VORLAX_induced_velocity = settings.use_VORLAX_matrix_calculation
    if not use_VORLAX_induced_velocity:
        A =   jnp.multiply(C_mn[:,:,:,0],jnp.atleast_3d(jnp.sin(delta)*jnp.cos(phi))) \
            + jnp.multiply(C_mn[:,:,:,1],jnp.atleast_3d(jnp.cos(delta)*jnp.sin(phi))) \
            - jnp.multiply(C_mn[:,:,:,2],jnp.atleast_3d(jnp.cos(phi)*jnp.cos(delta)))   # validated from book eqn 7.42 
    else:
        A = EW

    # Compute vortex strength
    GAMMA  = jnp.linalg.solve(A,RHS)

    # ---------------------------------------------------------------------------------------
    # STEP 11: Compute Pressure Coefficient
    # ------------------ --------------------------------------------------------------------   
    #VORLAX subroutine = PRESS
                  
    # spanwise strip exposure flag, always 0 for SUAVE's infinitely thin airfoils. Needs to change if thick airfoils added
    RJTS = 0                         
    
    # COMPUTE FREE-STREAM AND ONSET FLOW PARAMETERS. Used throughout the remainder of VLM
    B2     = jnp.tile((mach**2 - 1),n_cp)
    SINALF = jnp.sin(aoa)
    COSALF = jnp.cos(aoa)
    SINPSI = jnp.sin(PSI)
    COPSI  = jnp.cos(PSI)
    COSIN  = COSALF *SINPSI *2.0
    COSINP = COSALF *SINPSI
    COSCOS = COSALF *COPSI
    PITCH  = PITCHQ /VINF
    ROLL   = ROLLQ /VINF
    YAW    = YAWQ /VINF    
    
    # reshape CHORD
    CHORD  = CHORD[0,:]
    CHORD_strip = CHORD[LE_ind]

    # COMPUTE EFFECT OF SIDESLIP on DCP intermediate variables. needs change if cosine chorwise spacing added
    FORAXL = COSCOS
    FORLAT = COSIN
    
    TAN_LE = (XB1[LE_ind] - XA1[LE_ind])/ \
                jnp.sqrt((ZB1[LE_ind]-ZA1[LE_ind])**2 + \
                        (YB1[LE_ind]-YA1[LE_ind])**2)  
    TAN_TE = (XB_TE - XA_TE)/ jnp.sqrt((ZB_TE-ZA_TE)**2 + (YB_TE-YA_TE)**2) # _TE variables already have np.repeat built in 
    TAN_LE = jnp.broadcast_to(jnp.repeat(TAN_LE,RNMAX[LE_ind]),jnp.shape(B2)) 
    TAN_TE = jnp.broadcast_to(TAN_TE                         ,jnp.shape(B2))    
    
    TNL    = TAN_LE * 1 # VORLAX's SIGN variable not needed, as these are taken directly from geometry
    TNT    = TAN_TE * 1
    XIA    = jnp.broadcast_to((RK-1)/RNMAX, jnp.shape(B2))
    XIB    = jnp.broadcast_to((RK  )/RNMAX, jnp.shape(B2))
    TANA   = TNL *(1. - XIA) + TNT *XIA
    TANB   = TNL *(1. - XIB) + TNT *XIB
    
    # cumsum GANT loop if KTOP > 0 (don't actually need KTOP with vectorized arrays and np.roll)
    GFX    = jnp.tile((1 /CHORD), (len_mach,1))
    GANT   = strip_cumsum(GFX*GAMMA, chord_breaks, RNMAX[LE_ind])
    GANT   = jnp.roll(GANT,1)
    GANT   = w(LE_ind[na,:],0,GANT) 
    
    GLAT   = GANT *(TANA - TANB) - GFX *GAMMA *TANB
    COS_DL = (YBH-YAH)[LE_ind]/D
    cos_DL = jnp.broadcast_to(jnp.repeat(COS_DL,RNMAX[LE_ind]),jnp.shape(B2))
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
    SURF = jnp.array(VD.wing_areas)
    SREF = Sref  

    # Flip coordinates on the other side of the wing
    boolean = YBH<0. 
    XA1, XB1= w(boolean,XB1,XA1), w(boolean,XA1,XB1)
    YAH, YBH= w(boolean,YBH,YAH), w(boolean,YAH,YBH)

    # Leading edge sweep. VORLAX does it panel by panel. This will be spanwise.
    TLE   = TAN_LE[:,LE_ind]
    B2_LE = B2[:,LE_ind]
    T2    = TLE*TLE
    STB   = jnp.zeros_like(B2_LE)
    STB   = w(B2_LE<T2,np.sqrt(T2-B2_LE),STB)
    
    # DL IS THE DIHEDRAL ANGLE (WITH RESPECT TO THE X-Y PLANE) OF
    # THE IR STREAMWISE STRIP OF HORSESHOE VORTICES. 
    COD = jnp.cos(phi[0,LE_ind])  # Just the LE values
    
    SID = jnp.sin(phi[0,LE_ind])  # Just the LE values

    # Now on to each strip
    PION = 2.0 /RNMAX
    ADC  = 0.5*PION

    # XLE = LOCATION OF FIRST VORTEX MIDPOINT IN FRACTION OF CHORD.
    XLE = 0.125 *PION
    
    GAF = 0.5 + 0.5 *RJTS**2

    # CORMED IS LENGTH OF STRIP CENTERLINE BETWEEN LOAD POINT
    # AND TRAILING EDGE THIS PARAMETER IS USED IN THE COMPUTATION
    # OF THE STRIP ROLLING COUPLE CONTRIBUTION DUE TO SIDESLIP.
    X      = XCH                       #x-coord of load point (horseshoe centroid)
    XTE    = (VD.XA_TE + VD.XB_TE)/2   #Trailing edge x-coord behind the control point  
    CORMED = XTE - X   

    # SINF REFERENCES THE LOAD CONTRIBUTION OF IRT-VORTEX TO THE
    # STRIP NOMINAL AREA, I.E., AREA OF STRIP ASSUMING CONSTANT
    # (CHORDWISE) HORSESHOE SPAN.    
    SINF = ADC * DCP # The horshoe span lengths have been removed since VST/VSS == 1 always

    # Split into chordwise strengths and sum into strips    
    # SICPLE = COUPLE (ABOUT STRIP CENTERLINE) DUE TO SIDESLIP.
    CNC    = np.add.reduceat(SINF       ,chord_breaks,axis=1)
    SICPLE = np.add.reduceat(SINF*CORMED,chord_breaks,axis=1)

    # COMPUTE SLOPE (TX) WITH RESPECT TO X-AXIS AT LOAD POINTS BY INTER
    # POLATING BETWEEN CONTROL POINTS AND TAKING INTO ACCOUNT THE LOCAL
    # INCIDENCE.    
    XX   = (RK - .75) *PION /2.0
    TX    = SLOPE - ZETA
    CAXL  = -SINF*TX/(1.0+TX**2) # These are the axial forces on each panel
    BMLE  = (XLE-XX)*SINF        # These are moment on each panel
    
    # Sum onto the panel
    CAXL = np.add.reduceat(CAXL,chord_breaks,axis=1)
    BMLE = np.add.reduceat(BMLE,chord_breaks,axis=1)
    
    SICPLE *= (-1) * COSIN * COD * GAF
    DCP_LE = DCP[:,LE_ind]
    
    # COMPUTE LEADING EDGE THRUST COEFF. (CSUC) BY CALCULATING
    # THE TOTAL INDUCED FLOW AT THE LEADING EDGE. THIS COMPUTATION
    # ONLY PERFORMED FOR COSINE CHORDWISE SPACING (LAX = 0).    
    # ** TO DO ** Add cosine spacing (earlier in VLM) to properly capture the magnitude of these earlier.
    # Right now, this computation still happens with linear spacing, though its effects are underestimated.
    CLE = compute_rotation_effects(VD, settings, EW, GAMMA, len_mach, X, CHORD, XLE, XBAR, 
                                   rhs, COSINP, SINALF, PITCH, ROLL, YAW, STB, RNMAX)    
    
    # Leading edge suction multiplier. See documentation. This is a negative integer if used
    # Default to 1 unless specified otherwise
    SPC  = K_SPC*jnp.ones_like(DCP_LE)
    
    # If the vehicle is subsonic and there is vortex lift enabled then SPC changes to -1
    VL       = jnp.repeat(jnp.array(VD.vortex_lift),n_sw)
    m_b      = jnp.atleast_2d(mach[:,0]<1.)
    SPC_cond = VL*m_b.T
    SPC      = w(SPC_cond,-1.,SPC)
    SPC      = SPC * exposed_leading_edge_flag
    
    CLE      = CLE + 0.5* DCP_LE *jnp.sqrt(XLE[LE_ind])
    CSUC     = 0.5*jnp.pi*jnp.abs(SPC)*(CLE**2)*STB 

    # TFX AND TFZ ARE THE COMPONENTS OF LEADING EDGE FORCE VECTOR ALONG
    # ALONG THE X AND Z BODY AXES.   
    
    SLE  = SLOPE[LE_ind]
    ZETA = ZETA[LE_ind]
    XCOS = jnp.broadcast_to(jnp.cos(SLE-ZETA),jnp.shape(DCP_LE))
    XSIN = jnp.broadcast_to(jnp.sin(SLE-ZETA),jnp.shape(DCP_LE))
    TFX  =  1.*XCOS
    TFZ  = -1.*XSIN

    # If a negative number is used for SPC a different correction is used. See VORLAX documentation for Lan reference
    TFX  = w(SPC<0,XSIN*jnp.sign(DCP_LE),TFX)
    TFZ  = w(SPC<0,jnp.abs(XCOS)*jnp.sign(DCP_LE),TFZ)

    CAXL = CAXL - TFX*CSUC
    
    # Add a dimension into the suction to be chordwise
    CNC   = CNC + CSUC*jnp.sqrt(1+T2)*TFZ
    
    # FCOS AND FSIN ARE THE COSINE AND SINE OF THE ANGLE BETWEEN
    # THE CHORDLINE OF THE IR-STRIP AND THE X-AXIS    
    FCOS = jnp.cos(ZETA)
    FSIN = jnp.sin(ZETA)
    
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
    X      = VD.XCH[LE_ind]  # These are all LE values
    Y      = VD.YCH[LE_ind]  # These are all LE values
    Z      = VD.ZCH[LE_ind]  # These are all LE values
    BMX    = BFZ * Y - BFY * (Z - ZBAR)
    BMX    = BMX + SICPLE
    BMY    = BMLE * COD + BFX * (Z - ZBAR) - BFZ * (X - XBAR)
    BMZ    = BMLE * SID - BFX * Y + BFY * (X - XBAR)
    CDC    = BFZ * SINALF +  (BFX *COPSI + BFY *SINPSI) * COSALF
    CDC    = CDC * CHORD_strip 

    ES     = 2*s[0,LE_ind]
    STRIP  = ES *CHORD_strip
    LIFT   = (BFZ *COSALF - (BFX *COPSI + BFY *SINPSI) *SINALF)*STRIP   
    DRAG   = CDC*ES 
    MOMENT = STRIP * (BMY *COPSI - BMX *SINPSI)  
    FY     = (BFY *COPSI - BFX *SINPSI) *STRIP
    RM     = STRIP *(BMX *COSALF *COPSI + BMY *COSALF *SINPSI + BMZ *SINALF)
    YM     = STRIP *(BMZ *COSALF - (BMX *COPSI + BMY *SINPSI) *SINALF)

    # Now calculate the coefficients for each wing
    cl_y     = LIFT/CHORD_strip/ES
    cdi_y    = DRAG/CHORD_strip/ES
    CL_wing  = np.add.reduceat(LIFT,span_breaks,axis=1)/SURF
    CDi_wing = np.add.reduceat(DRAG,span_breaks,axis=1)/SURF
    alpha_i  = jnp.hsplit(jnp.arctan(cdi_y/cl_y),span_breaks[1:])
    
    # Now calculate total coefficients
    CL       = jnp.atleast_2d(jnp.sum(LIFT,axis=1)/SREF).T          # CLTOT in VORLAX
    CDi      = jnp.atleast_2d(jnp.sum(DRAG,axis=1)/SREF).T          # CDTOT in VORLAX
    CM       = jnp.atleast_2d(jnp.sum(MOMENT,axis=1)/SREF).T/c_bar  # CMTOT in VORLAX
    CYTOT    = jnp.atleast_2d(jnp.sum(FY,axis=1)/SREF).T   # total y force coeff
    CRTOT    = jnp.atleast_2d(jnp.sum(RM,axis=1)/SREF).T   # rolling moment coeff (unscaled)
    CRMTOT   = CRTOT/w_span*(-1)                         # rolling moment coeff
    CNTOT    = jnp.atleast_2d(jnp.sum(YM,axis=1)/SREF).T   # yawing  moment coeff (unscaled)
    CYMTOT   = CNTOT/w_span*(-1)                         # yawing  moment coeff

    # ---------------------------------------------------------------------------------------
    # STEP 13: Pack outputs
    # ------------------ --------------------------------------------------------------------     
    precision      = settings.floating_point_precision
    
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
    results.CL_wing        = CL_wing   
    results.CDi_wing       = CDi_wing 
    results.cl_y           = cl_y   
    results.cdi_y          = cdi_y       
    results.alpha_i        = alpha_i  
    results.CP             = jnp.array(CP    , dtype=precision)
    results.gamma          = jnp.array(GAMMA , dtype=precision)
    results.VD             = VD
    results.V_distribution = rhs.V_distribution
    results.V_x            = rhs.Vx_ind_total
    results.V_z            = rhs.Vz_ind_total
    
    return results

# ----------------------------------------------------------------------
#  CLE rotation effects helper function
# ----------------------------------------------------------------------
def compute_rotation_effects(VD, settings, EW_small, GAMMA, len_mach, X, CHORD, XLE, XBAR, 
                             rhs, COSINP, SINALF, PITCH, ROLL, YAW, STB, RNMAX):
    """ This computes the effects of the freestream and aircraft rotation rate on 
    CLE, the induced flow at the leading edge
    
    Assumptions:
    Several of the values needed in this calculation have been computed earlier and stored in VD
    
    Normally, VORLAX skips the calculation implemented in this function for linear 
    chordwise spacing (the if statement below). However, since the trends are correct, 
    albeit underestimated, this calculation is being forced here.    
    """
    LE_ind      = VD.leading_edge_indices
    RNMAX       = VD.panels_per_strip

    ##spacing = settings.spanwise_cosine_spacing
    ##if spacing == False: # linear spacing is LAX==1 in VORLAX
    ##    return 0 #CLE not calculated till later for linear spacing
    
    # Computate rotational effects (pitch, roll, yaw rates) on LE suction
    # pick leading edge strip values for EW and reshape GAMMA -> gamma accordingly
    EW    = EW_small[: ,LE_ind, :]
    n_tot_strips = EW.shape[1]
    gamma = jnp.array(jnp.split(jnp.repeat(GAMMA, n_tot_strips, axis=0), len_mach))
    CLE = (EW*gamma).sum(axis=2)
    
    # Up till EFFINC, some of the following values were computed in compute_RHS_matrix().
    #     EFFINC and ALOC are calculated the exact same way, except for the XGIRO term.
    # LOCATE VORTEX LATTICE CONTROL POINT WITH RESPECT TO THE
    # ROTATION CENTER (XBAR, 0, ZBAR). THE RELATIVE COORDINATES
    # ARE XGIRO, YGIRO, AND ZGIRO. 
    XGIRO = X - CHORD*XLE - jnp.repeat(XBAR, RNMAX[LE_ind])
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
    CLE = CLE - EFFINC[:,LE_ind] 
    CLE = jnp.where(STB > 0, CLE /RNMAX[LE_ind] /STB, CLE)
    
    return CLE

# ----------------------------------------------------------------------
#  Vectorized cumsum from indices
# ----------------------------------------------------------------------
def strip_cumsum(arr, chord_breaks, strip_lengths):
    """ Uses numpy to to compute a cumsum that resets along
    the leading edge of every strip.
    
    Assumptions:
    chordwise_breaks always starts at 0
    """    
    cumsum  = jnp.cumsum(arr, axis=1)
    offsets = cumsum[:,chord_breaks-1]
    offsets.at[:,0].set(0)
    offsets = jnp.repeat(offsets, strip_lengths, axis=1)
    return cumsum - offsets