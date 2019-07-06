## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# VLM.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Apr 2017, T. MacDonald
#           Oct 2017, E. Botero
#           Jun 2018, M. Clarke


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_induced_velocity_matrix import  compute_induced_velocity_matrix
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_vortex_distribution import compute_vortex_distribution

# ----------------------------------------------------------------------
#  Weissinger Vortex Lattice
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
    n_sw   = settings.number_panels_spanwise    
    n_cw   = settings.number_panels_chordwise   
    Sref   = geometry.reference_area
    
    # define point about which moment coefficient is computed 
    c_bar  = geometry.wings['main_wing'].chords.mean_aerodynamic
    x_mac  = geometry.wings['main_wing'].aerodynamic_center[0] + geometry.wings['main_wing'].origin[0]
    x_cg   = geometry.mass_properties.center_of_gravity[0] 
    if x_cg == None:
        x_m = x_mac 
    else:
        x_m = x_cg
    
    aoa  = conditions.aerodynamics.angle_of_attack   # angle of attack  
    mach = conditions.freestream.mach_number*0         # mach number
    ones = np.atleast_2d(np.ones_like(aoa)) 
   
    # generate vortex distribution
    VD = compute_vortex_distribution(geometry,settings)  
    
    # Build induced velocity matrix, C_mn
    C_mn, DW_mn = compute_induced_velocity_matrix(VD,n_sw,n_cw,aoa,mach)
    MCM = VD.MCM 
    
    # Compute flow tangency conditions   
    inv_root_beta = 1/np.sqrt(1-(mach**2))     
    inv_root_beta[np.isnan(inv_root_beta)] = 1.0
    inv_root_beta = np.atleast_2d(inv_root_beta)
    
    phi   = np.arctan((VD.ZBC - VD.ZAC)/(VD.YBC - VD.YAC))*ones
    delta = np.arctan((VD.ZC - VD.ZCH)/((VD.XC - VD.XCH)*inv_root_beta))   
   
    # Build Aerodynamic Influence Coefficient Matrix
    A = C_mn[:,:,:,2] - np.multiply(C_mn[:,:,:,0],np.atleast_3d(np.tan(delta)))- np.multiply(C_mn[:,:,:,1],np.atleast_3d(np.tan(phi)))  # EDIT
    
    # Build the vector
    RHS = np.tan(delta)*np.cos(aoa) - np.sin(aoa)
    
    # Compute vortex strength  
    gamma = np.linalg.solve(A,RHS)
    
    # Do some matrix magic
    len_aoa = len(aoa)
    len_cps = VD.n_cp
    eye = np.eye(len_aoa)
    tile_eye = np.broadcast_to(eye,(len_cps,len_aoa,len_aoa))
    tile_eye =  np.transpose(tile_eye,axes=[1,0,2])
    
    # Compute induced velocities     
    u = np.dot(C_mn[:,:,:,0]*MCM[:,:,:,0],gamma[:,:].T)[:,:,0]
    v = np.dot(C_mn[:,:,:,1]*MCM[:,:,:,1],gamma[:,:].T)[:,:,0]
    w = np.sum(np.dot(C_mn[:,:,:,2]*MCM[:,:,:,2],gamma[:,:].T)*tile_eye,axis=2)
    w_ind = np.sum(np.dot(DW_mn[:,:,:,2],gamma[:,:].T)*tile_eye,axis=2)
    
    # ---------------------------------------------------------------------------------------
    # STEP 10: Compute aerodynamic coefficients 
    # --------------------------------------------------------------------------------------- 
    n_cp       = VD.n_cp   
    n_cppw     = n_sw*n_cw
    n_w        = VD.n_w
    CS         = VD.CS*ones
    wing_areas = VD.wing_areas
    X_M        = np.ones(n_cp)*x_m  *ones
    CL_wing    = np.zeros(n_w)
    CDi_wing   = np.zeros(n_w)
    
    Del_Y = np.abs(VD.YB1 - VD.YA1)*ones
    
    # Linspace out where breaks are
    wing_space = np.linspace(0,n_cppw*n_w,n_w+1)
    
    # Use split to divide u, w, gamma, and Del_y into more arrays
    u_n_w        = np.array(np.array_split(u,n_w,axis=1))
    u_n_w_sw     = np.array(np.array_split(u,n_w*n_sw,axis=1))
    w_n_w        = np.array(np.array_split(w,n_w,axis=1))
    w_n_w_sw     = np.array(np.array_split(w,n_w*n_sw,axis=1))
    w_ind_n_w    = np.array(np.array_split(w_ind,n_w,axis=1))
    w_ind_n_w_sw = np.array(np.array_split(w_ind,n_w*n_sw,axis=1))    
    gamma_n_w    = np.array(np.array_split(gamma,n_w,axis=1))
    gamma_n_w_sw = np.array(np.array_split(gamma,n_w*n_sw,axis=1))
    Del_Y_n_w    = np.array(np.array_split(Del_Y,n_w,axis=1))
    Del_Y_n_w_sw = np.array(np.array_split(Del_Y,n_w*n_sw,axis=1))
    
    # Calculate the Coefficients on each wing individually
    L_wing   = np.sum(np.multiply(u_n_w+1,(gamma_n_w*Del_Y_n_w)),axis=2).T
    CL_wing  = L_wing/wing_areas
    Di_wing  = np.sum(np.multiply(-w_ind_n_w,(gamma_n_w*Del_Y_n_w)),axis=2).T
    CDi_wing = Di_wing/wing_areas
    
    # Calculate each spanwise set of Cls and Cds
    cl_y = np.sum(np.multiply(u_n_w_sw +1,(gamma_n_w_sw*Del_Y_n_w_sw)),axis=2).T/CS
    cdi_y = np.sum(np.multiply(-w_ind_n_w_sw,(gamma_n_w_sw*Del_Y_n_w_sw)),axis=2).T/CS
    
    # Split the Cls and Cds for each wing
    Cl_wings = np.array(np.split(cl_y,n_w,axis=1))
    Cd_wings = np.array(np.split(cdi_y,n_w,axis=1))
            
    # total lift and lift coefficient
    L  = np.atleast_2d(np.sum(np.multiply((1+u),gamma*Del_Y),axis=1)).T
    CL = 2*L/(Sref) 
    
    # total drag and drag coefficient
    D  =  -np.atleast_2d(np.sum(np.multiply(w_ind,gamma*Del_Y),axis=1)).T
    CDi = D/(2*Sref)   

    # pressure coefficient
    U_tot = np.sqrt((1+u)*(1+u) + v*v + w*w)
    CP = 1 - (U_tot)*(U_tot)
     
    # moment coefficient
    CM  = np.atleast_2d(np.sum(np.multiply((X_M - VD.XCH*ones),Del_Y*gamma),axis=1)/(Sref*c_bar)).T     
    
    return CL, CDi, CM, CL_wing, CDi_wing, cl_y , cdi_y , CP 