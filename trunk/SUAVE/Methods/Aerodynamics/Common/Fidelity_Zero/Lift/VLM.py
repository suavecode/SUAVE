## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# VLM.py
# 
# Created:  May 2019, M. Clarke
#           Jul 2020, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_induced_velocity_matrix import compute_induced_velocity_matrix
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_vortex_distribution     import compute_vortex_distribution
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_RHS_matrix              import compute_RHS_matrix
# ----------------------------------------------------------------------
#  Vortex Lattice
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def VLM(conditions,settings,geometry):
    """Uses the vortex lattice method to compute the lift, induced drag and moment coefficients  

    Assumptions:
    None

    Source:
    1. Aerodynamics for Engineers, Sixth Edition by John Bertin & Russel Cummings 
    Pgs. 379-397(Literature)
    
    2. Low-Speed Aerodynamics, Second Edition by Joseph katz, Allen Plotkin
    Pgs. 331-338(Literature), 579-586 (Fortran Code implementation)
    
    3. Yahyaoui, M. "Generalized Vortex Lattice Method for Predicting Characteristics of Wings
    with Flap and Aileron Deflection" , World Academy of Science, Engineering and Technology 
    International Journal of Mechanical, Aerospace, Industrial and Mechatronics Engineering 
    Vol:8 No:10, 2014
    

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
        
       settings.number_panels_spanwise         [Unitless]
       settings.number_panels_chordwise        [Unitless]
       settings.use_surrogate                  [Unitless]
       settings.include_slipstream_effect      [Unitless]
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
    n_sw       = settings.number_panels_spanwise    
    n_cw       = settings.number_panels_chordwise   
    sur_flag   = settings.use_surrogate
    slipstream = settings.include_slipstream_effect
    Sref       = geometry.reference_area
    
    
    # define point about which moment coefficient is computed
    if 'main_wing' in geometry.wings:
        c_bar      = geometry.wings['main_wing'].chords.mean_aerodynamic
        x_mac      = geometry.wings['main_wing'].aerodynamic_center[0] + geometry.wings['main_wing'].origin[0][0]
    else:
        c_bar = 0.
        x_mac = 0.
        for wing in geometry.wings:
            if wing.vertical == False:
                if c_bar <= wing.chords.mean_aerodynamic:
                    c_bar = wing.chords.mean_aerodynamic
                    x_mac = wing.aerodynamic_center[0] + wing.origin[0][0]
            
    x_cg       = geometry.mass_properties.center_of_gravity[0][0]
    if x_cg == None:
        x_m = x_mac 
    else:
        x_m = x_cg
    
    aoa  = conditions.aerodynamics.angle_of_attack   # angle of attack  
    mach = conditions.freestream.mach_number         # mach number
    ones = np.atleast_2d(np.ones_like(aoa)) 
   
    # generate vortex distribution
    VD = compute_vortex_distribution(geometry,settings)  
    
    # Build induced velocity matrix, C_mn
    C_mn, DW_mn  = compute_induced_velocity_matrix(VD,n_sw,n_cw,aoa,mach)
    MCM = VD.MCM 
    
    # Compute flow tangency conditions   
    inv_root_beta = np.zeros_like(mach)
    inv_root_beta[mach<1] = 1/np.sqrt(1-mach[mach<1]**2)     
    inv_root_beta[mach>1] = 1/np.sqrt(mach[mach>1]**2-1) 
    if np.any(mach==1):
        raise('Mach of 1 cannot be used in building compressibility corrections.')
    inv_root_beta = np.atleast_2d(inv_root_beta)
    
    phi   = np.arctan((VD.ZBC - VD.ZAC)/(VD.YBC - VD.YAC))*ones          # dihedral angle 
    delta = np.arctan((VD.ZC - VD.ZCH)/((VD.XC - VD.XCH)*inv_root_beta)) # mean camber surface angle 
   
    # Build Aerodynamic Influence Coefficient Matrix
    A =   np.multiply(C_mn[:,:,:,0],np.atleast_3d(np.sin(delta)*np.cos(phi))) \
        + np.multiply(C_mn[:,:,:,1],np.atleast_3d(np.cos(delta)*np.sin(phi))) \
        - np.multiply(C_mn[:,:,:,2],np.atleast_3d(np.cos(phi)*np.cos(delta)))   # valdiated from book eqn 7.42 
    
    B =   np.multiply(DW_mn[:,:,:,0],np.atleast_3d(np.sin(delta)*np.cos(phi))) \
        + np.multiply(DW_mn[:,:,:,1],np.atleast_3d(np.cos(delta)*np.sin(phi))) \
        - np.multiply(DW_mn[:,:,:,2],np.atleast_3d(np.cos(phi)*np.cos(delta)))   # valdiated from book eqn 7.42     
   
   
    # Build the vector
    RHS = compute_RHS_matrix(n_sw,n_cw,delta,phi,conditions,geometry,sur_flag,slipstream)

    # Compute vortex strength  
    n_cp  = VD.n_cp  
    gamma = np.linalg.solve(A,RHS)
    gamma_3d = np.repeat(np.atleast_3d(gamma), n_cp ,axis = 2 )
    u = np.sum(C_mn[:,:,:,0]*MCM[:,:,:,0]*gamma_3d, axis = 2) 
    v = np.sum(C_mn[:,:,:,1]*MCM[:,:,:,1]*gamma_3d, axis = 2) 
    w = np.sum(C_mn[:,:,:,2]*MCM[:,:,:,2]*gamma_3d, axis = 2) 
    w_ind = -np.sum(B*MCM[:,:,:,2]*gamma_3d, axis = 2) 
     
    # ---------------------------------------------------------------------------------------
    # STEP 10: Compute aerodynamic coefficients 
    # ---------------------------------------------------------------------------------------  
    n_w        = VD.n_w
    CS         = VD.CS*ones
    wing_areas = np.array(VD.wing_areas)
    X_M        = np.ones(n_cp)*x_m  *ones
    CL_wing    = np.zeros(n_w)
    CDi_wing   = np.zeros(n_w)
    
    Del_Y = np.abs(VD.YB1 - VD.YA1)*ones 
    
    # Use split to divide u, w, gamma, and Del_y into more arrays
    u_n_w        = np.array(np.array_split(u,n_w,axis=1))
    u_n_w_sw     = np.array(np.array_split(u,n_w*n_sw,axis=1)) 
    w_ind_n_w    = np.array(np.array_split(w_ind,n_w,axis=1))
    w_ind_n_w_sw = np.array(np.array_split(w_ind,n_w*n_sw,axis=1))    
    gamma_n_w    = np.array(np.array_split(gamma,n_w,axis=1))
    gamma_n_w_sw = np.array(np.array_split(gamma,n_w*n_sw,axis=1))
    Del_Y_n_w    = np.array(np.array_split(Del_Y,n_w,axis=1))
    Del_Y_n_w_sw = np.array(np.array_split(Del_Y,n_w*n_sw,axis=1)) 
    
    # lift coefficients on each wing   
    machw             = np.tile(mach,len(wing_areas))     
    L_wing            = np.sum(np.multiply(u_n_w+1,(gamma_n_w*Del_Y_n_w)),axis=2).T
    CL_wing           = L_wing/(0.5*wing_areas)
    CL_wing[machw>1]  = CL_wing[machw>1]*8 # supersonic lift off by a factor of 8 
    
    # drag coefficients on each wing  
    Di_wing           = np.sum(np.multiply(-w_ind_n_w,(gamma_n_w*Del_Y_n_w)),axis=2).T
    CDi_wing          = Di_wing/(0.5*wing_areas)
    CDi_wing[machw>1] = CDi_wing[machw>1]*2   # supersonic drag off by a factor of 2 
    
    # Calculate each spanwise set of Cls and Cds
    cl_y        = np.sum(np.multiply(u_n_w_sw +1,(gamma_n_w_sw*Del_Y_n_w_sw)),axis=2).T/CS
    cdi_y       = np.sum(np.multiply(-w_ind_n_w_sw,(gamma_n_w_sw*Del_Y_n_w_sw)),axis=2).T/CS 
            
    # total lift and lift coefficient
    L           = np.atleast_2d(np.sum(np.multiply((1+u),gamma*Del_Y),axis=1)).T 
    CL          = L/(0.5*Sref)           # validated form page 402-404, aerodynamics for engineers # supersonic lift off by 2^3 
    CL[mach>1]  = CL[mach>1]*8   # supersonic lift off by a factor of 8 
    
    # total drag and drag coefficient
    D           =   -np.atleast_2d(np.sum(np.multiply(w_ind,gamma*Del_Y),axis=1)).T   
    CDi         = D/(0.5*Sref)  
    CDi[mach>1] = CDi[mach>1]*2 # supersonic drag off by a factor of 2 
    
    # pressure coefficient
    U_tot       = np.sqrt((1+u)*(1+u) + v*v + w*w)
    CP          = 1 - (U_tot)*(U_tot)
     
    # moment coefficient
    CM          = np.atleast_2d(np.sum(np.multiply((X_M - VD.XCH*ones),Del_Y*gamma),axis=1)/(Sref*c_bar)).T     
    
    # delete MCM from VD data structure since it consumes memory
    delattr(VD, 'MCM')   
    
    return CL, CDi, CM, CL_wing, CDi_wing, cl_y , cdi_y , CP 