## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
# generate_fidelity_one_wake_shape.py
#
# Created:  Jan 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data
from SUAVE.Methods.Propulsion.Rotor_Wake.Fidelity_Zero.compute_wake_contraction_matrix import compute_wake_contraction_matrix


# package imports
import numpy as np


## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def generate_fidelity_one_wake_shape(wake,rotor):
    """
    This generates the propeller wake control points and vortex distribution that make up the prescribed vortex wake. 
    All (x,y,z) coordinates are in the vehicle frame of reference (X points nose to tail).
    
    Assumptions:
       None
    
    Source: 
       None
    
    Inputs:
       rotor  -  A SUAVE rotor component for which the wake is generated
    
    """
        
    # Unpack rotor
    R                = rotor.tip_radius
    r                = rotor.radius_distribution 
    c                = rotor.chord_distribution 
    beta             = rotor.twist_distribution + rotor.inputs.pitch_command
    B                = rotor.number_of_blades  
    
    rotor_outputs    = rotor.outputs
    Na               = rotor_outputs.number_azimuthal_stations
    Nr               = rotor_outputs.number_radial_stations

    omega            = rotor_outputs.omega                               
    va               = rotor_outputs.disc_axial_induced_velocity
    V_inf            = rotor_outputs.velocity
    gamma            = rotor_outputs.disc_circulation
    rot              = rotor.rotation
    
    # apply rotation direction to twist and chord distribution
    c    = -rot*c
    beta = -rot*beta
    
    # dimensions for analysis                      
    Nr   = len(r)                   # number of radial stations
    m    = len(omega)               # number of control points

    # Compute blade angles starting from each of Na azimuthal stations, shape: (Na,B)
    azi          = np.linspace(0,2*np.pi,Na+1)[:-1]
    azi_initial  = np.atleast_2d(np.linspace(0,2*np.pi,B+1)[:-1])
    blade_angles = (azi_initial + np.atleast_2d(azi).T) 
    
    # Extract specified wake settings:
    init_timestep_offset = wake.wake_settings.initial_timestep_offset
    n_rotations          = wake.wake_settings.number_rotor_rotations
    tsteps_per_rot       = wake.wake_settings.number_steps_per_rotation
    
    # Calculate additional wake properties
    dt    = (azi[1]-azi[0])/omega[0][0]
    nts   = tsteps_per_rot*n_rotations
    
    # Compute properties for each wake timestep
    ts                = np.linspace(0,dt*(nts-1),nts) 
    omega_ts          = np.multiply(omega,np.atleast_2d(ts))  # Angle of each azimuthal station in nts
    
    # Update start angle of rotor
    t0                = dt*init_timestep_offset
    start_angle       = omega[0]*t0 
    rotor.start_angle = start_angle[0]
    
    # extract mean inflow velocities
    axial_induced_velocity = np.mean(va,axis = 2) # radial inflow, averaged around the azimuth
    mean_induced_velocity  = np.mean( axial_induced_velocity,axis = 1)   
    
    rots = rotor.body_to_prop_vel()[0]
    
    lambda_tot   = np.atleast_2d((np.dot(V_inf,rots[0])  + mean_induced_velocity)).T /(omega*R)   # inflow advance ratio (page 99 Leishman)
    mu_prop      = np.atleast_2d(np.dot(V_inf,rots[2])).T /(omega*R)                              # rotor advance ratio  (page 99 Leishman) 
    Vx           = np.repeat(V_inf[:,0,None], Nr, axis=1) # shape: (m,Nr)
    Vz           = np.repeat(V_inf[:,2,None], Nr, axis=1) # shape: (m,Nr)
    V_prop       = np.sqrt((Vx  + axial_induced_velocity)**2 + Vz**2)

    # wake skew angle 
    wake_skew_angle = -(np.arctan(mu_prop/lambda_tot))
    wake_skew_angle = np.tile(wake_skew_angle[:,:,None],(1,Nr,nts))
    
    # reshape gamma to find the average between stations           
    gamma_new = (gamma[:,:-1,:] + gamma[:,1:,:])*0.5  # [control points, Nr-1, Na ] one less radial station because ring
    
    num       = Na//B
    time_idx  = np.arange(nts)
    Gamma     = np.zeros((Na,m,B,Nr-1,nts))
    
    # generate Gamma for each start angle
    for ito in range(Na):
        t_idx     = np.atleast_2d(time_idx).T 
        B_idx     = np.arange(B) 
        B_loc     = (ito + B_idx*num - t_idx )%Na 
        Gamma1    = gamma_new[:,:,B_loc]  
        Gamma1    = Gamma1.transpose(0,3,1,2) 
        Gamma[ito,:,:,:,:] = Gamma1
  
    # --------------------------------------------------------------------------------------------------------------
    #    ( control point , blade number , radial location on blade , time step )
    # --------------------------------------------------------------------------------------------------------------
    V_p = np.repeat(V_prop[:,:,None],len(ts),axis=2)
                    
    sx_inf0            = np.multiply(V_p*np.cos(wake_skew_angle), np.repeat(np.atleast_2d(ts)[:,None,:],Nr,axis=1))
    sx_inf             = np.tile(sx_inf0[None,:, None, :,:], (Na,1,B,1,1))
                      
    sy_inf0            = np.multiply(np.atleast_2d(V_inf[:,1]).T,np.atleast_2d(ts)) # = zero since no crosswind
    sy_inf             = -rot*np.tile(sy_inf0[None,:, None, None,:], (Na,1,B,Nr,1)) 
    
    sz_inf0            = np.multiply(V_p*np.sin(wake_skew_angle),np.repeat(np.atleast_2d(ts)[:,None,:],Nr,axis=1))
    sz_inf             = np.tile(sz_inf0[None,:, None, :,:], (Na,1,B,1,1))        
    
    # wake panel and blade angles
    start_angle_offset = np.tile(start_angle[None,:,None,None,None], (Na,1,B,Nr,nts))
    blade_angle_loc    = start_angle_offset + np.tile( blade_angles[:,None,:,None,None], (1,m,1,Nr,nts))  # negative rotation, positive blade angle location
    
    # offset angle of trailing wake panels relative to blade location
    total_angle_offset = np.tile(omega_ts[None,:,None,None,:], (Na,1,B,Nr,1))   
    
    # azimuthal position of each wake panel, (blade start index, ctrl_pts, B, Nr, nts)
    panel_azimuthal_positions = rot*(total_angle_offset - blade_angle_loc)      # axial view in rotor frame (angle 0 aligned with z-axis); 
    
    # put into velocity frame and find (y,z) components
    azi_y   = np.sin(panel_azimuthal_positions)
    azi_z   = np.cos(panel_azimuthal_positions) 
   
    # trailing edge points in airfoil coordinates 
    airfoils = rotor.Airfoils
    af       = airfoils[list(airfoils.keys())[0]]
    a_loc    = rotor.airfoil_polar_stations
    xupper   = np.zeros((Nr,len(af.geometry.x_upper_surface)))
    yupper   = np.zeros((Nr,len(af.geometry.x_upper_surface)))
    for i,airfoil in enumerate(airfoils):
        a_geo        = airfoil.geometry
        locs         = np.where(np.array(a_loc) == i )
        xupper[locs] = a_geo.x_upper_surface 
        yupper[locs] = a_geo.y_upper_surface 
    
    # Align the quarter chords of the airfoils (zero sweep)
    airfoil_le_offset = -c/2
    xte_airfoils      = xupper[:,-1]*c + airfoil_le_offset
    yte_airfoils      = yupper[:,-1]*c  
    xle_airfoils      = xupper[:,0]*c + airfoil_le_offset
    yle_airfoils      = yupper[:,0]*c  
    x_c_4_airfoils    = (xle_airfoils - xte_airfoils)/4 - airfoil_le_offset
    y_c_4_airfoils    = (yle_airfoils - yte_airfoils)/4
    
    # apply blade twist rotation along rotor radius
    xte_twisted = np.cos(beta)*xte_airfoils - np.sin(beta)*yte_airfoils        
    yte_twisted = np.sin(beta)*xte_airfoils + np.cos(beta)*yte_airfoils    
    
    x_c_4_twisted = np.cos(beta)*x_c_4_airfoils - np.sin(beta)*y_c_4_airfoils 
    y_c_4_twisted = np.sin(beta)*x_c_4_airfoils + np.cos(beta)*y_c_4_airfoils  
    
    # transform coordinates from airfoil frame to rotor frame
    xte = np.tile(np.atleast_2d(yte_twisted), (B,1))
    xte_rotor = np.tile(xte[None,:,:,None], (m,1,1,nts))  
    yte_rotor = -np.tile(xte_twisted[None,None,:,None],(m,B,1,1))*np.cos(panel_azimuthal_positions)
    zte_rotor = np.tile(xte_twisted[None,None,:,None],(m,B,1,1))*np.sin(panel_azimuthal_positions)
    
    r_4d = np.tile(r[None,None,:,None], (m,B,1,nts))
    
    x0 = 0
    y0 = r_4d*azi_y
    z0 = r_4d*azi_z
    
    x_pts0 = x0 + xte_rotor
    y_pts0 = y0 + yte_rotor
    z_pts0 = z0 + zte_rotor
    
    x_c_4_rotor = x0 - np.tile(y_c_4_twisted[None,None,:,None], (m,B,1,nts))
    y_c_4_rotor = y0 + np.tile(x_c_4_twisted[None,None,:,None], (m,B,1,nts))*np.cos(panel_azimuthal_positions)
    z_c_4_rotor = z0 - np.tile(x_c_4_twisted[None,None,:,None], (m,B,1,nts))*np.sin(panel_azimuthal_positions)   
    
    # compute wake contraction, apply to y-z plane
    X_pts0           = x_pts0 + sx_inf
    wake_contraction = compute_wake_contraction_matrix(rotor,Nr,m,nts,X_pts0,rotor_outputs) 
    Y_pts0           = y_pts0*wake_contraction + sy_inf
    Z_pts0           = z_pts0*wake_contraction + sz_inf
    
    # append propeller wake to each of its repeated origins  
    X_pts   = rotor.origin[0][0] + X_pts0  
    Y_pts   = rotor.origin[0][1] + Y_pts0
    Z_pts   = rotor.origin[0][2] + Z_pts0

    #------------------------------------------------------     
    # Account for lifting line panels
    #------------------------------------------------------
    x_c_4 = np.repeat(x_c_4_rotor[None,:,:,:,:], Na, axis=0) + rotor.origin[0][0]
    y_c_4 = (y_c_4_rotor) + rotor.origin[0][1]
    z_c_4 = (z_c_4_rotor) + rotor.origin[0][2]
    
    # prepend points at quarter chord to account for rotor lifting line
    X_pts = np.append(x_c_4[:,:,:,:,0][:,:,:,:,None], X_pts, axis=4) 
    Y_pts = np.append(y_c_4[:,:,:,:,0][:,:,:,:,None], Y_pts, axis=4)
    Z_pts = np.append(z_c_4[:,:,:,:,0][:,:,:,:,None], Z_pts, axis=4)
    
    #------------------------------------------------------
    # Store points  
    #------------------------------------------------------
    # Initialize vortex distribution and arrays with required matrix sizes
    VD = Data()
    rotor.vortex_distribution = VD        
    VD = initialize_distributions(Nr, Na, B, nts, m,VD)
    
    # ( azimuthal start index, control point  , blade number , location on blade, time step )
    if rot==-1:
        # panels ordered root to tip, A for inner-most panel edge
        VD.Wake.XA1[:,:,0:B,:,:] = X_pts[:, : , :, :-1 , :-1 ]
        VD.Wake.YA1[:,:,0:B,:,:] = Y_pts[:, : , :, :-1 , :-1 ]
        VD.Wake.ZA1[:,:,0:B,:,:] = Z_pts[:, : , :, :-1 , :-1 ]
        VD.Wake.XA2[:,:,0:B,:,:] = X_pts[:, : , :, :-1 ,  1: ]
        VD.Wake.YA2[:,:,0:B,:,:] = Y_pts[:, : , :, :-1 ,  1: ]
        VD.Wake.ZA2[:,:,0:B,:,:] = Z_pts[:, : , :, :-1 ,  1: ]
        VD.Wake.XB1[:,:,0:B,:,:] = X_pts[:, : , :, 1:  , :-1 ]
        VD.Wake.YB1[:,:,0:B,:,:] = Y_pts[:, : , :, 1:  , :-1 ]
        VD.Wake.ZB1[:,:,0:B,:,:] = Z_pts[:, : , :, 1:  , :-1 ]
        VD.Wake.XB2[:,:,0:B,:,:] = X_pts[:, : , :, 1:  ,  1: ]
        VD.Wake.YB2[:,:,0:B,:,:] = Y_pts[:, : , :, 1:  ,  1: ]
        VD.Wake.ZB2[:,:,0:B,:,:] = Z_pts[:, : , :, 1:  ,  1: ] 
    else:            
        # positive rotation reverses the A,B nomenclature of the panel
        VD.Wake.XA1[:,:,0:B,:,:] = X_pts[:, : , :, 1: , :-1 ]
        VD.Wake.YA1[:,:,0:B,:,:] = Y_pts[:, : , :, 1: , :-1 ]
        VD.Wake.ZA1[:,:,0:B,:,:] = Z_pts[:, : , :, 1: , :-1 ]
        VD.Wake.XA2[:,:,0:B,:,:] = X_pts[:, : , :, 1: ,  1: ]
        VD.Wake.YA2[:,:,0:B,:,:] = Y_pts[:, : , :, 1: ,  1: ]
        VD.Wake.ZA2[:,:,0:B,:,:] = Z_pts[:, : , :, 1: ,  1: ]
        VD.Wake.XB1[:,:,0:B,:,:] = X_pts[:, : , :, :-1  , :-1 ]
        VD.Wake.YB1[:,:,0:B,:,:] = Y_pts[:, : , :, :-1  , :-1 ]
        VD.Wake.ZB1[:,:,0:B,:,:] = Z_pts[:, : , :, :-1  , :-1 ]
        VD.Wake.XB2[:,:,0:B,:,:] = X_pts[:, : , :, :-1  ,  1: ]
        VD.Wake.YB2[:,:,0:B,:,:] = Y_pts[:, : , :, :-1  ,  1: ]
        VD.Wake.ZB2[:,:,0:B,:,:] = Z_pts[:, : , :, :-1  ,  1: ] 
        

    VD.Wake.GAMMA[:,:,0:B,:,:] = Gamma 
    
    # Append wake geometry and vortex strengths to each individual propeller
    wake.vortex_distribution.reshaped_wake   = VD.Wake
    
    # append trailing edge locations
    wake.vortex_distribution.reshaped_wake.Xblades_te = X_pts[:,0,:,:,0]
    wake.vortex_distribution.reshaped_wake.Yblades_te = Y_pts[:,0,:,:,0]
    wake.vortex_distribution.reshaped_wake.Zblades_te = Z_pts[:,0,:,:,0]

    # append quarter chord lifting line point locations        
    wake.vortex_distribution.reshaped_wake.Xblades_c_4 = x_c_4_rotor
    wake.vortex_distribution.reshaped_wake.Yblades_c_4 = y_c_4_rotor
    wake.vortex_distribution.reshaped_wake.Zblades_c_4 = z_c_4_rotor
    
    # append three-quarter chord evaluation point locations        
    wake.vortex_distribution.reshaped_wake.Xblades_cp = x_c_4 
    wake.vortex_distribution.reshaped_wake.Yblades_cp = y_c_4 
    wake.vortex_distribution.reshaped_wake.Zblades_cp = z_c_4 

    # Compress Data into 1D Arrays  
    mat6_size = (Na,m,nts*B*(Nr-1)) 

    wake.vortex_distribution.XA1    =  np.reshape(VD.Wake.XA1,mat6_size)
    wake.vortex_distribution.YA1    =  np.reshape(VD.Wake.YA1,mat6_size)
    wake.vortex_distribution.ZA1    =  np.reshape(VD.Wake.ZA1,mat6_size)
    wake.vortex_distribution.XA2    =  np.reshape(VD.Wake.XA2,mat6_size)
    wake.vortex_distribution.YA2    =  np.reshape(VD.Wake.YA2,mat6_size)
    wake.vortex_distribution.ZA2    =  np.reshape(VD.Wake.ZA2,mat6_size)
    wake.vortex_distribution.XB1    =  np.reshape(VD.Wake.XB1,mat6_size)
    wake.vortex_distribution.YB1    =  np.reshape(VD.Wake.YB1,mat6_size)
    wake.vortex_distribution.ZB1    =  np.reshape(VD.Wake.ZB1,mat6_size)
    wake.vortex_distribution.XB2    =  np.reshape(VD.Wake.XB2,mat6_size)
    wake.vortex_distribution.YB2    =  np.reshape(VD.Wake.YB2,mat6_size)
    wake.vortex_distribution.ZB2    =  np.reshape(VD.Wake.ZB2,mat6_size)
    wake.vortex_distribution.GAMMA  =  np.reshape(VD.Wake.GAMMA,mat6_size)
    
    rotor.wake_skew_angle = wake_skew_angle
    
    return wake, rotor

## @ingroup Methods-Propulsion-Rotor_Wake-Fidelity_One
def initialize_distributions(Nr, Na, B, n_wts, m, VD):
    """
    Initializes the matrices for the wake vortex distributions.
    
    Assumptions:
        None

    Source:
        N/A
        
    Inputs:
       Nr    - number of radial blade elemnts
       Na    - number of azimuthal start positions
       B     - number of rotor blades
       n_wts - total number of wake time steps in wake simulation
       m     - number of control points to evaluate
       VD    - vehicle vortex distribution
       
    Outputs:
       VD  - Vortex distribution
       WD  - Wake vortex distribution
    
    Properties:
       N/A
       
    """
    nmax = Nr - 1 # one less vortex ring than blade elements
    
    VD.Wake       = Data()
    mat1_size     = (Na,m,B,nmax,n_wts)
    VD.Wake.XA1   = np.zeros(mat1_size) 
    VD.Wake.YA1   = np.zeros(mat1_size) 
    VD.Wake.ZA1   = np.zeros(mat1_size) 
    VD.Wake.XA2   = np.zeros(mat1_size) 
    VD.Wake.YA2   = np.zeros(mat1_size) 
    VD.Wake.ZA2   = np.zeros(mat1_size)    
    VD.Wake.XB1   = np.zeros(mat1_size) 
    VD.Wake.YB1   = np.zeros(mat1_size) 
    VD.Wake.ZB1   = np.zeros(mat1_size) 
    VD.Wake.XB2   = np.zeros(mat1_size) 
    VD.Wake.YB2   = np.zeros(mat1_size) 
    VD.Wake.ZB2   = np.zeros(mat1_size) 
    VD.Wake.GAMMA  = np.zeros(mat1_size)  
 
    return VD