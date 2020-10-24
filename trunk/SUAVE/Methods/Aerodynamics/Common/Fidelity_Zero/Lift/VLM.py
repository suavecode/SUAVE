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
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_wing_vortex_distribution  import generate_wing_vortex_distribution
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_RHS_matrix                 import compute_RHS_matrix 

# ----------------------------------------------------------------------
#  Vortex Lattice
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift 
def VLM(conditions,settings,geometry,initial_timestep_offset = 0 ,wake_development_time = 0.05 ):
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
    VD   = generate_wing_vortex_distribution(geometry,settings)   
    
    # pack vortex distribution 
    geometry.vortex_distribution = VD
    
    # Build induced velocity matrix, C_mn
    C_mn, DW_mn = compute_wing_induced_velocity(VD,n_sw,n_cw,aoa,mach) 
     
    # Compute flow tangency conditions   
    inv_root_beta           = np.zeros_like(mach)
    inv_root_beta[mach<1]   = 1/np.sqrt(1-mach[mach<1]**2)     
    inv_root_beta[mach>1]   = 1/np.sqrt(mach[mach>1]**2-1)   
    inv_root_beta           = np.atleast_2d(inv_root_beta)
    
    phi   = np.arctan((VD.ZBC - VD.ZAC)/(VD.YBC - VD.YAC))*ones          # dihedral angle 
    delta = np.arctan((VD.ZC - VD.ZCH)/((VD.XC - VD.XCH)*inv_root_beta)) # mean camber surface angle 
   
    # Build Aerodynamic Influence Coefficient Matrix
    A =   np.multiply(C_mn[:,:,:,0],np.atleast_3d(np.sin(delta)*np.cos(phi))) \
        + np.multiply(C_mn[:,:,:,1],np.atleast_3d(np.cos(delta)*np.sin(phi))) \
        - np.multiply(C_mn[:,:,:,2],np.atleast_3d(np.cos(phi)*np.cos(delta)))   # valdiated from book eqn 7.42  
   
    # Build the vector
    RHS  ,Vx_ind_total , Vz_ind_total , V_distribution , dt = compute_RHS_matrix(n_sw,n_cw,delta,phi,conditions,geometry,\
                                                                                 pwm,initial_timestep_offset,wake_development_time ) 
    
    # Spersonic Vortex Lattice - Validated from NASA CR, page 3 
    locs = np.where(mach>1)[0]
    DW_mn[locs]  = DW_mn[locs]*2
    C_mn[locs]   = C_mn[locs]*2
    
    # Compute vortex strength  
    n_cp     = VD.n_cp  
    gamma    = np.linalg.solve(A,RHS)
    gamma_3d = np.repeat(np.atleast_3d(gamma), n_cp ,axis = 2 )
    u        = np.sum(C_mn[:,:,:,0]*gamma_3d, axis = 2)  
    w_ind    = -np.sum(DW_mn[:,:,:,2]*gamma_3d, axis = 2) 
     
    # ---------------------------------------------------------------------------------------
    # STEP 10: Compute aerodynamic coefficients 
    # ------------------ ---------------------------------------------------------------------  
    n_w        = VD.n_w
    CS         = VD.CS*ones
    CS_w       = np.array(np.array_split(CS,n_w,axis=1)) 
    wing_areas = np.array(VD.wing_areas)
    X_M        = np.ones(n_cp)*x_m  *ones
    CL_wing    = np.zeros(n_w)
    CDi_wing   = np.zeros(n_w) 
    Del_Y      = np.abs(VD.YB1 - VD.YA1)*ones  
    
    # Use split to divide u, w, gamma, and Del_y into more arrays
    u_n_w        = np.array(np.array_split(u,n_w,axis=1))  
    w_ind_n_w_sw = np.array(np.array_split(w_ind,n_w*n_sw,axis=1))    
    gamma_n_w    = np.array(np.array_split(gamma,n_w,axis=1))
    gamma_n_w_sw = np.array(np.array_split(gamma,n_w*n_sw,axis=1))
    Del_Y_n_w    = np.array(np.array_split(Del_Y,n_w,axis=1))
    Del_Y_n_w_sw = np.array(np.array_split(Del_Y,n_w*n_sw,axis=1)) 
    
    # --------------------------------------------------------------------------------------------------------
    # LIFT                                                                          
    # --------------------------------------------------------------------------------------------------------    
    # lift coefficients on each wing   
    machw             = np.tile(mach,len(wing_areas))     
    L_wing            = np.sum(np.multiply(u_n_w+1,(gamma_n_w*Del_Y_n_w)),axis=2).T
    CL_wing           = L_wing/(0.5*wing_areas)
    
    # Calculate spanwise lift 
    spanwise_Del_y    = Del_Y_n_w_sw[:,:,0]
    spanwise_Del_y_w  = np.array(np.array_split(Del_Y_n_w_sw[:,:,0].T,n_w,axis = 1))
    
    cl_y              = (2*(np.sum(gamma_n_w_sw,axis=2)*spanwise_Del_y).T)/CS
    cl_y_w            = np.array(np.array_split(cl_y ,n_w,axis=1)) 
    
    # total lift and lift coefficient
    L                 = np.atleast_2d(np.sum(np.multiply((1+u),gamma*Del_Y),axis=1)).T 
    CL                = L/(0.5*Sref)   # validated form page 402-404, aerodynamics for engineers
    
    # --------------------------------------------------------------------------------------------------------
    # DRAG                                                                          
    # --------------------------------------------------------------------------------------------------------         
    # drag coefficients on each wing   
    w_ind_sw_w        = np.array(np.array_split(np.sum(w_ind_n_w_sw,axis = 2).T ,n_w,axis = 1))
    Di_wing           = np.sum(w_ind_sw_w*spanwise_Del_y_w*cl_y_w*CS_w,axis = 2) 
    CDi_wing          = Di_wing.T/(wing_areas)  
    
    # total drag and drag coefficient 
    spanwise_w_ind    = np.sum(w_ind_n_w_sw,axis=2).T    
    D                 = np.sum(spanwise_w_ind*spanwise_Del_y.T*cl_y*CS,axis = 1) 
    cdi_y             = spanwise_w_ind*spanwise_Del_y.T*cl_y*CS
    CDi               = np.atleast_2d(D/(Sref)).T  
    
    # --------------------------------------------------------------------------------------------------------
    # PRESSURE                                                                      
    # --------------------------------------------------------------------------------------------------------          
    L_ij              = np.multiply((1+u),gamma*Del_Y) 
    CP                = L_ij/VD.panel_areas  
    
    # --------------------------------------------------------------------------------------------------------
    # MOMENT                                                                        
    # --------------------------------------------------------------------------------------------------------             
    CM                = np.atleast_2d(np.sum(np.multiply((X_M - VD.XCH*ones),Del_Y*gamma),axis=1)/(Sref*c_bar)).T     
    
    Velocity_Profile = Data()
    Velocity_Profile.Vx_ind   = Vx_ind_total
    Velocity_Profile.Vz_ind   = Vz_ind_total
    Velocity_Profile.V        = V_distribution 
    Velocity_Profile.dt       = dt 
    
    return CL, CDi, CM, CL_wing, CDi_wing, cl_y , cdi_y , CP ,Velocity_Profile