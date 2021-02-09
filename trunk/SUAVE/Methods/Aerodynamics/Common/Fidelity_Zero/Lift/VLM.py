## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# VLM.py
# 
# Created:  May 2019, M. Clarke
#           Jul 2020, E. Botero
#           Sep 2020, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
import numpy as np 
from SUAVE.Core import Data
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wing_induced_velocity      import compute_wing_induced_velocity
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wing_induced_velocity_sup  import compute_wing_induced_velocity_sup
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_wing_vortex_distribution  import generate_wing_vortex_distribution, compute_panel_area, compute_unit_normal
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_RHS_matrix                 import compute_RHS_matrix 

# ----------------------------------------------------------------------
#  Vortex Lattice
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift 
def VLM(conditions,settings,geometry,initial_timestep_offset = 0 ,wake_development_time = 0.05 ):
    """Uses the vortex lattice method to compute the lift, induced drag and moment coefficients  

    Assumptions: None

    Source:
    1. Aerodynamics for Engineers, Sixth Edition by John Bertin & Russel Cummings 
    Pgs. 379-397(Literature)
    
    2. Low-Speed Aerodynamics, Second Edition by Joseph katz, Allen Plotkin
    Pgs. 331-338(Literature), 579-586 (Fortran Code implementation)
    
    3. Yahyaoui, M. "Generalized Vortex Lattice Method for Predicting Characteristics of Wings
    with Flap and Aileron Deflection" , World Academy of Science, Engineering and Technology 
    International Journal of Mechanical, Aerospace, Industrial and Mechatronics Engineering 
    Vol:8 No:10, 2014
    
    4. Miranda, Luis R., Robert D. Elliot, and William M. Baker. "A generalized vortex 
    lattice method for subsonic and supersonic flow applications." (1977). (NASA CR)

    Inputs:
    geometry.
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
        
       settings.number_spanwise_vortices         [Unitless]
       settings.number_chordwise_vortices        [Unitless]
       settings.use_surrogate                  [Unitless]
       settings.propeller_wake_model           [Unitless]
       conditions.aerodynamics.angle_of_attack [radians]
       conditions.freestream.mach_number       [Unitless]
       
    Outputs:                                   
    CL                                         [Unitless]
    Cl                                         [Unitless]
    CDi                                        [Unitless]
    Cdi                                        [Unitless]

    Properties Used:
    N/A
    """ 
   
    # unpack settings
    n_sw       = settings.number_spanwise_vortices    
    n_cw       = settings.number_chordwise_vortices 
    pwm        = settings.propeller_wake_model
    ito        = settings.initial_timestep_offset
    wdt        = settings.wake_development_time    
    Sref       = geometry.reference_area 
    

    # define point about which moment coefficient is computed
    if 'main_wing' in geometry.wings:
        c_bar      = geometry.wings['main_wing'].chords.mean_aerodynamic
        x_mac      = geometry.wings['main_wing'].aerodynamic_center[0] + geometry.wings['main_wing'].origin[0][0]
        z_mac      = geometry.wings['main_wing'].aerodynamic_center[2] + geometry.wings['main_wing'].origin[0][2]
    else:
        c_bar = 0.
        x_mac = 0.
        for wing in geometry.wings:
            if wing.vertical == False:
                if c_bar <= wing.chords.mean_aerodynamic:
                    c_bar = wing.chords.mean_aerodynamic
                    x_mac = wing.aerodynamic_center[0] + wing.origin[0][0]
            
    x_cg       = geometry.mass_properties.center_of_gravity[0][0]
    z_cg       = geometry.mass_properties.center_of_gravity[0][2]
    if x_cg == None:
        x_m = x_mac 
        z_m = z_mac
    else:
        x_m = x_cg
        z_m = z_cg
    
    aoa  = conditions.aerodynamics.angle_of_attack   # angle of attack  
    mach = conditions.freestream.mach_number         # mach number
    ones = np.atleast_2d(np.ones_like(mach)) 
   
    # generate vortex distribution 
    VD   = generate_wing_vortex_distribution(geometry,settings)  
    n_w  = VD.n_w    
    n_cp = VD.n_cp  
    
    # pack vortex distribution 
    geometry.vortex_distribution = VD
    
    # Build induced velocity matrix, C_mn
    C_mn, s, t, CHORD, RFLAG = compute_wing_induced_velocity_sup(VD,n_sw,n_cw,aoa,mach) 
     
    # Compute flow tangency conditions   
    inv_root_beta           = np.zeros_like(mach)
    inv_root_beta[mach<1]   = 1/np.sqrt(1-mach[mach<1]**2)     
    inv_root_beta[mach>1]   = 1.#1/np.sqrt(mach[mach>1]**2-1)   
    inv_root_beta           = np.atleast_2d(inv_root_beta)
    
    phi   = np.arctan((VD.ZBC - VD.ZAC)/(VD.YBC - VD.YAC))*ones          # dihedral angle 
    delta = np.arctan((VD.ZC - VD.ZCH)/((VD.XC - VD.XCH)*inv_root_beta)) # mean camber surface angle 
   
    # Build Aerodynamic Influence Coefficient Matrix
    A =   np.multiply(C_mn[:,:,:,0],np.atleast_3d(np.sin(delta)*np.cos(phi))) \
        + np.multiply(C_mn[:,:,:,1],np.atleast_3d(np.cos(delta)*np.sin(phi))) \
        - np.multiply(C_mn[:,:,:,2],np.atleast_3d(np.cos(phi)*np.cos(delta)))   # validated from book eqn 7.42  
   
    # Build the vector
    RHS  ,Vx_ind_total , Vz_ind_total , V_distribution , dt = compute_RHS_matrix(n_sw,n_cw,delta,phi,conditions,geometry,\
                                                                                     pwm,ito,wdt )
    # Turn off sonic vortices when Mach>1
    RHS = RHS*RFLAG
    
    # Compute vortex strength  
    gamma = np.linalg.solve(A,RHS)
    
    ## ---------------------------------------------------------------------------------------
    ## STEP 10: Compute Pressure Coefficient
    ## ------------------ --------------------------------------------------------------------

    # COMPUTE FREE-STREAM AND ONSET FLOW PARAMETERS. If yaw is ever added these equations would change
    B2     = np.tile((mach**2 - 1),n_cp)
    COSALF = np.tile(np.cos(aoa),n_cp)
    FORAXL = COSALF
    
    FLAX   = 1.0 # Assume linear spaced panels in the chordwise direction (LAX)
    
    RNMAX   = n_cw*1.
    FACTOR  = FORAXL
    CHORD   = CHORD[:,0,:]
    t       = t[:,0,:]
    
    # COMPUTE LOAD COEFFICIENT
    GNET = gamma*FACTOR*RNMAX/CHORD
    DCP  = 2*GNET
    CP   = DCP
    
    # ---------------------------------------------------------------------------------------
    # STEP 11: Compute aerodynamic coefficients 
    # ------------------ -------------------------------------------------------------------- 
    
    # PSI is 0 for zero yaw, we also are doing zero roll and no pitch rate
    SINALF  = np.tile(np.sin(aoa),n_cp)
    COSALF  = np.tile(np.cos(aoa),n_cp)
    SINPSI  = 0.
    COPSI   = 1.
    
    # Work panel by panel
    SURF = np.array(VD.wing_areas)
    SREF = Sref

    # Leading edge sweep and trailing edge sweep. VORLAX does it panel by panel. This will be spanwise.
    YBH = VD.YBH*ones
    XA1 = VD.XA1*ones
    XB1 = VD.XB1*ones
    ZA1 = VD.ZA1*ones
    ZB1 = VD.ZB1*ones    
    
    # This is only valid for calculating LE sweeps
    boolean = YBH<0. 
    XA1[boolean], XB1[boolean] = XB1[boolean], XA1[boolean]
        
    TLE = t[:,0::n_cw]
    TLE = np.repeat(TLE,n_cw,axis=1)
    T2  = TLE**2
    STB = np.zeros_like(u)
    STB[B2<T2] = np.sqrt(T2[B2<T2]-B2[B2<T2])
    STB = STB[:,0::n_cw]
    
    # Panel Dihedral Angle, using AH and BH location
    YAH = VD.YAH*ones
    ZAH = VD.ZAH*ones
    ZBH = VD.ZBH*ones
    
    YAH[boolean], YBH[boolean] = YBH[boolean], YAH[boolean]
    
    D   = np.sqrt((YAH-YBH)**2+(ZAH-ZBH)**2)
    
    SID = ((ZBH-ZAH)/D)[:,0::n_cw] # Just the LE values
    COD = ((YBH-YAH)/D)[:,0::n_cw] # Just the LE values
    
    # Now on to each strip
    PION = (np.pi *(1.0 - FLAX) + 2.0 *FLAX) /RNMAX
    ADC  = 0.5*PION
    JTS  = 0.
    
    # XLE = LOCATION OF FIRST VORTEX MIDPOINT IN FRACTION OF CHORD.
    XLE =.5 *(1.0 -np.cos (.5 *PION)) *(1.0 - FLAX) + 0.125 *PION *FLAX
    
    SICPLE = 0.0 #  COUPLE (ABOUT STRIP CENTERLINE) DUE TO SIDESLIP (no sideslip)
    
    # SINF REFERENCES THE LOAD CONTRIBUTION OF IRT-VORTEX TO THE
    # STRIP NOMINAL AREA, I.E., AREA OF STRIP ASSUMING CONSTANT
    # (CHORDWISE) HORSESHOE SPAN.    
    VST = s[:,0,:]
    VSS = s[:,0,:]
    
    SINF = ADC * DCP *VST /VSS
    
    # Split into chordwise strengths and sum into strips
    CNC = np.array(np.split(np.reshape(SINF,(-1,n_cw)).sum(axis=1),len(mach)))
    
    # COMPUTE SLOPE (TX) WITH RESPECT TO X-AXIS AT LOAD POINTS BY INTER
    # POLATING BETWEEN CONTROL POINTS AND TAKING INTO ACCOUNT THE LOCAL
    # INCIDENCE.    
    RK   = np.tile(np.linspace(1,n_cw,n_cw),n_sw*n_w)*ones
    IRT  = np.tile(np.linspace(1,n_sw,n_sw),n_cw*n_w)*ones
    XX   = .5 *(1. - np.cos ((RK - .5) *PION))
    XX   = XX *(1.0 - FLAX) + (RK - .75) *PION /2.0 *FLAX
    K    = 1*RK
    KX   = 1*K
    IRTX = 1*IRT
    KX[K>1]   = K[K>1]-1
    IRTX[K>1] = IRT[[K>1]] - 1
    
    # The exact results of IRTX will not match VORLAX because of indexing differences in python
    RKX = KX
    X1  = .5 *(1. - np.cos (RKX *PION))
    X1  = X1 *(1.0 - FLAX) + (RKX - .25) *PION /2.0 *FLAX
    X2  = .5 *(1. - np.cos((RKX + 1.) *PION))
    X2  = X2 *(1.0 - FLAX) + (RKX + .75) *PION /2.0 *FLAX
    
    # CHECK ME LATER WITH CAMBER!!!
    XA2 = VD.XA2*ones
    XB2 = VD.XB2*ones
    ZA1 = VD.ZA1*ones
    ZB1 = VD.ZB1*ones       
    ZA2 = VD.ZA2*ones
    ZB2 = VD.ZB2*ones
    
    X1c  = (XA1+XB1)/2
    X2c  = (XA2+XB2)/2
    Z1c  = (ZA1+ZB1)/2
    Z2c  = (ZA2+ZB2)/2
    
    SLOPE = (Z2c-Z1c)/(X2c-X1c)
    
    # This section takes differences for F1 and F2 based on the slopes
    all_aft_indices = np.linspace(1,(n_w*n_cw*n_sw),(n_w*n_cw*n_sw),dtype=int)
    all_for_indices = all_aft_indices-1
    
    mask1 = np.ones_like(all_aft_indices,dtype=bool)
    mask1[n_cw-1::n_cw] =False
    
    aft_indices = all_aft_indices[mask1]
    for_indices = all_for_indices[mask1]
    
    F1 = SLOPE
    F1[:,aft_indices] = SLOPE[:,for_indices]

    F2 = np.zeros_like(SLOPE)
    F2[:,for_indices] = SLOPE[:,aft_indices]
    
    # Now fill in the LE values for F2
    mask2 = np.zeros_like(all_aft_indices,dtype=bool)
    mask3 = np.zeros_like(all_aft_indices,dtype=bool)
    
    mask2[0::n_cw] = True
    mask3[1::n_cw] = True
    
    F2[:,all_for_indices[mask2]] = SLOPE[:,all_for_indices[mask3]] 
    
    # Zeta needs updating!
    ZETA = 0.
    
    TANX = (XX-X2)/(X1-X2)*F1 +(XX-X1)/(X2-X1)*F2
    TX   = TANX - ZETA
    CAXL = -SINF*TX/(1.0+TX**2) # These are the axial forces on each panel
    BMLE = (XLE-XX)*SINF        # These are moment on each panel
    
    # Sum onto the panel
    CAXL = np.array(np.split(np.reshape(CAXL,(-1,n_cw)).sum(axis=1),len(mach)))
    BMLE = np.array(np.split(np.reshape(BMLE,(-1,n_cw)).sum(axis=1),len(mach)))
    
    XX = XLE
    
    SPC    = 1. # Leading edge suction multiplier. See documentation. This is a negative integer if used
    DCP_LE = DCP[:,0::n_cw]
    CLE    = 0.5* DCP_LE *np.sqrt(XX)*FLAX
    CSUC   = 0.5*np.pi*np.abs(SPC)*(CLE**2)*STB
    
    # SLE is slope at leading edge
    SLE  = SLOPE[:,0::n_cw]
    FKEY = 1. - JTS*(1+JTS)
    XCOS = 1./np.sqrt(1+(SLE-ZETA)**2)
    XSIN = (SLE-ZETA)*XCOS
    TFX  = XCOS
    TFZ  = - XSIN
    
    if SPC<0.:
        TFX = XSIN*np.sign(DCP[:,0::n_cw])*FKEY
        TFZ = np.abs(XCOS)*np.sign(DCP[:,0::n_cw])*FKEY
        
    CAXL = CAXL -TFX*CSUC

    # Add a dimension into the suction to be chordwise
    T2_LE = T2[:,0::n_cw]
    CNC   = CNC + CSUC*np.sqrt(1+T2_LE)*TFZ
    
    # FCOS AND FSIN ARE THE COSINE AND SINE OF THE ANGLE BETWEEN
    # THE CHORDLINE OF THE IR-STRIP AND THE X-AXIS    
    FCOS = 1./np.sqrt(1.+ ZETA*ZETA)
    FSIN = FCOS*ZETA
    
    # BFX, BFY, AND BFZ ARE THE COMPONENTS ALONG THE BODY AXES
    # OF THE STRIP FORCE CONTRIBUTION.
    BFX = - CNC *FSIN + CAXL *FCOS
    BFY = - (CNC *FCOS + CAXL *FSIN) *SID
    BFZ = (CNC *FCOS + CAXL *FSIN) *COD
    
    # CONVERT CNC FROM CN INTO CNC (COEFF. *CHORD).
    CHORD_strip = CHORD[:,0::n_cw]
    CNC  = CNC  * CHORD_strip
    BMLE = BMLE * CHORD_strip
    
    # BMX, BMY, AND BMZ ARE THE COMPONENTS ALONG THE BODY AXES
    # OF THE STRIP MOMENT (ABOUT MOM. REF. POINT) CONTRIBUTION.
    X      = ((VD.XAH+VD.XBH)/2)[0::n_cw]  # These are all LE values
    Y      = ((VD.YAH+VD.YBH)/2)[0::n_cw]  # These are all LE values
    Z      = ((VD.ZAH+VD.ZBH)/2)[0::n_cw]  # These are all LE values
    SINALF = SINALF[:,0::n_cw]
    COSALF = COSALF[:,0::n_cw]   
    XBAR   = np.ones(n_sw*n_w) * x_m
    ZBAR   = np.ones(n_sw*n_w) * z_m
    BMX    = BFZ * Y - BFY * (Z - ZBAR)
    BMX    = BMX + SICPLE
    BMY    = BMLE * COD + BFX * (Z - ZBAR) - BFZ * (X - XBAR)
    CDC    = BFZ * SINALF +  (BFX *COPSI + BFY *SINPSI) *COSALF
    CDC    = CDC * CHORD_strip
    
    ES    = 2*s[:,0,:]
    ES    = ES[:,0::n_cw]
    STRIP = ES *CHORD_strip
    LIFT  = (BFZ *COSALF - (BFX *COPSI + BFY *SINPSI) *SINALF)*STRIP
    DRAG  = CDC*ES 
    
    MOMENT = STRIP *(BMY *COPSI - BMX *SINPSI)
    
    # Now calculate the coefficients for each wing and in total
    cl_y     = LIFT/CHORD_strip/ES
    cdi_y    = DRAG/CHORD_strip/ES
    CL_wing  = np.array(np.split(np.reshape(LIFT,(-1,n_sw)).sum(axis=1),len(mach)))/SURF
    CDi_wing = np.array(np.split(np.reshape(DRAG,(-1,n_sw)).sum(axis=1),len(mach)))/SURF
    CL       = np.atleast_2d(np.sum(LIFT,axis=1)/SREF).T
    CDi      = np.atleast_2d(np.sum(DRAG,axis=1)/SREF).T
    CM       = np.atleast_2d(np.sum(MOMENT,axis=1)/SREF).T/c_bar

    Velocity_Profile = Data()
    Velocity_Profile.Vx_ind   = Vx_ind_total
    Velocity_Profile.Vz_ind   = Vz_ind_total
    Velocity_Profile.V        = V_distribution 
    Velocity_Profile.dt       = dt
    
    ## Trying to see if we can integrate pressures to get CL and CD
    #panel_areas = VD.panel_areas
    #normals     = VD.normals
    #normals     = np.broadcast_to(normals,(len(mach),n_cp,3))
    #wing_areas  = np.array(VD.wing_areas)
    
    #force_per_panel = np.atleast_3d(panel_areas*CP)
    #dir_f_panel     = force_per_panel*normals
    #cosalpha        = np.cos(aoa)
    #sinalpha        = np.sin(aoa)
    
    #lift_force_panel = dir_f_panel[:,:,2]*cosalpha + dir_f_panel[:,:,0]*sinalpha
    #drag_force_panel = dir_f_panel[:,:,2]*sinalpha + dir_f_panel[:,:,0]*cosalpha
    
    #lift_strip  = np.array(np.split(np.reshape(lift_force_panel,(-1,n_cw)).sum(axis=1),len(mach)))
    #drag_strip  = np.array(np.split(np.reshape(drag_force_panel,(-1,n_cw)).sum(axis=1),len(mach)))
    #chord_strip = np.array(np.split(panel_areas,n_sw*n_w)).sum(axis=1) # area of a chordwise strip
    
    #cl_y     = lift_strip/chord_strip
    #cdi_y    = drag_strip/chord_strip
    #CL_wing  = np.array(np.split(np.reshape(lift_strip,(-1,n_sw)).sum(axis=1),len(mach)))/wing_areas
    #CDi_wing = np.array(np.split(np.reshape(drag_strip,(-1,n_sw)).sum(axis=1),len(mach)))/wing_areas
    #CL       = np.atleast_2d(np.sum(lift_strip,axis=1)/Sref).T
    #CDi      = np.atleast_2d(np.sum(drag_strip,axis=1)/Sref).T
        
    
    
    return CL, CDi, CM, CL_wing, CDi_wing, cl_y , cdi_y , CP ,Velocity_Profile
