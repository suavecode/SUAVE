## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# compute_source_coordinates.py
# 
# Created: Mar 2021, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
import numpy as np 
from SUAVE.Core import Data
# ----------------------------------------------------------------------
#  Source Coordinates 
# ---------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools 
def compute_point_source_coordinates(conditions,network,mls,source,settings):  
    """This calculated the position vector from a point source to the observer 
            
    Assumptions:
        N/A
    Source:
        N/A  
    Inputs:  
    conditions.
          
    network.
    
    mls                   - microphone locations       [m]
    Outputs: 
        position vector   - position vector of points  [m]
    Properties Used:
        N/A       
    """  
    
    
    # aquire dimension of matrix
    num_cpt         = conditions._size
    num_mic         = len(mls[0,:,0]) 
    
    if source == 'lift_rotors': 
        num_prop    = int(network.number_of_lift_rotor_engines)    
        prop_origin = []
        for prop in network.lift_rotors:
            prop_origin.append(prop.origin[0])
        prop_origin = np.array(prop_origin)
        
    else:
        num_prop    = int(network.number_of_propeller_engines)   
        prop_origin = []
        for prop in network.propellers:
            prop_origin.append(prop.origin[0])
        prop_origin = np.array(prop_origin)
        
    # Get the rotation matrix
    prop2body   = prop.prop_vel_to_body()
    

    # [control point, microphone , propeller , 2D geometry matrix ]
    # rotation of propeller about y axis by thurst angle (one extra dimension for translations)
    rotation_1                = np.zeros((num_cpt,num_mic,num_prop,4,4))
    rotation_1[:,:,:,0:3,0:3] = prop2body   
    rotation_1[:,:,:,3,3]     = 1     

    # translation to location on propeller
    I                         = np.atleast_3d(np.eye(4)).T
    translation_1             = np.repeat(np.repeat(np.repeat(I,num_prop, axis = 0)[np.newaxis,:,:,:],num_mic, axis = 0)[np.newaxis,:,:,:,:],num_cpt, axis = 0)
    translation_1[:,:,:,0,3]  = np.repeat(np.repeat(np.atleast_2d(prop_origin[:,0]),num_mic, axis = 0)[np.newaxis,:,:],num_cpt, axis = 0)     
    translation_1[:,:,:,1,3]  = np.repeat(np.repeat(np.atleast_2d(prop_origin[:,1]),num_mic, axis = 0)[np.newaxis,:,:],num_cpt, axis = 0)           
    translation_1[:,:,:,2,3]  = np.repeat(np.repeat(np.atleast_2d(prop_origin[:,2]),num_mic, axis = 0)[np.newaxis,:,:],num_cpt, axis = 0) 

    # rotation of vehicle about y axis by AoA 
    rotation_2                        = np.zeros((num_cpt,num_mic,num_prop,4,4))
    rotation_2[0:num_cpt,:,:,0:3,0:3] = conditions.frames.body.transform_to_inertial[:,np.newaxis,np.newaxis,:,:]
    rotation_2[:,:,:,3,3]     = 1 

    # translation of vehicle to air  
    translation_2               = np.repeat(np.repeat(np.repeat(I,num_prop, axis = 0)[np.newaxis,:,:,:],num_mic, axis = 0)[np.newaxis,:,:,:,:],num_cpt, axis = 0)       # control point, microphone, prop, 4x4
    translation_2[:,:,:,0,3]    = np.repeat(mls[:,:,0][:,:,np.newaxis],num_prop, axis = 2) 
    translation_2[:,:,:,1,3]    = np.repeat(mls[:,:,1][:,:,np.newaxis],num_prop, axis = 2) 
    translation_2[:,:,:,2,3]    = np.repeat(mls[:,:,2][:,:,np.newaxis],num_prop, axis = 2) 

    # identity transformation 
    I0    = np.atleast_3d(np.array([[0,0,0,1]]))
    I0   = np.array(I0)  
    mat_0 = np.repeat(np.repeat(np.repeat(I0,num_prop, axis = 0)[np.newaxis,:,:,:],num_mic, axis = 0)[np.newaxis,:,:,:,:],num_cpt, axis = 0)

    # execute operation  
    mat_1 = np.matmul(rotation_1,mat_0) 
    mat_2 = np.matmul(translation_1,mat_1)
    mat_3 = np.matmul(rotation_2,mat_2) 
    mat_4 = np.matmul(translation_2,mat_3)
    mat_4 = -mat_4

    # store points
    propeller_position_vector          = np.zeros((num_cpt,num_mic,num_prop,3))
    propeller_position_vector[:,:,:,0] = mat_4[:,:,:,0,0]
    propeller_position_vector[:,:,:,1] = mat_4[:,:,:,1,0]
    propeller_position_vector[:,:,:,2] = mat_4[:,:,:,2,0]
     
    return propeller_position_vector
 
## @ingroupMethods-Noise-Fidelity_One-Noise_Tools  
def compute_blade_section_source_coordinates(AoA,acoustic_outputs,network,mls,source,settings):  
    """This calculated the position vector from a point source to the observer 
            
    Assumptions:
        N/A
 
    Source:
        N/A  
 
    Inputs:  
    AoA                   - angle of attack           [rad]
    acoustic_outputs      
    network
    mls                   - microphone locations      [m] 
    source
    prop
    settings
    
 
    Outputs:  
 
    Properties Used:
        N/A       
    """
    
    # aquire dimension of matrix 
    num_cpt         = len(AoA)
    num_mic         = len(mls[0,:,0])   

    if source == 'lift_rotors': 
        propellers  = network.lift_rotors
        num_prop    = int(network.number_of_lift_rotor_engines)    
        prop_origin = []
        for prop in network.lift_rotors:
            prop_origin.append(prop.origin[0])
        prop_origin = np.array(prop_origin)
        
    else:
        propellers  = network.propellers
        num_prop    = int(network.number_of_propeller_engines)   
        prop_origin = []
        for prop in network.propellers:
            prop_origin.append(prop.origin[0])
        prop_origin = np.array(prop_origin) 
            
    prop          = propellers[list(propellers.keys())[0]]
    rots          = np.array(prop.orientation_euler_angles) * 1.
    thrust_angle  = rots[1] + prop.inputs.y_axis_rotation
    num_cf        = len(settings.center_frequencies)
    r             = prop.radius_distribution
    num_sec       = len(r)
    phi_2d0       = acoustic_outputs.disc_azimuthal_distribution 
    alpha_eff0    = acoustic_outputs.disc_effective_angle_of_attack
    num_azi       = len(phi_2d0[0,0,:])
    t_v0          = -AoA                                        # vehicle tilt angle between the vehicle hub plane and the geographical ground
    t_r0          = np.ones_like(AoA)*(np.pi/2 - thrust_angle)  # rotor tilt angle between the rotor hub plane and the vehicle hub plane

    # Update dimensions for computation   
    r                    = vectorize(r,num_cpt,num_mic,num_sec,num_prop,num_azi,num_cf,vectorize_method = 1) 
    sin_phi              = vectorize(np.sin(phi_2d0),num_cpt,num_mic,num_sec,num_prop,num_azi,num_cf,vectorize_method = 2) 
    cos_phi              = vectorize(np.cos(phi_2d0),num_cpt,num_mic,num_sec,num_prop,num_azi,num_cf,vectorize_method = 2) 
    sin_alpha_eff        = vectorize(np.sin(alpha_eff0),num_cpt,num_mic,num_sec,num_prop,num_azi,num_cf,vectorize_method = 2)
    cos_alpha_eff        = vectorize(np.cos(alpha_eff0),num_cpt,num_mic,num_sec,num_prop,num_azi,num_cf,vectorize_method = 2)
    cos_t_v              = vectorize(np.cos(t_v0),num_cpt,num_mic,num_sec,num_prop,num_azi,num_cf,vectorize_method = 3)
    sin_t_v              = vectorize(np.sin(t_v0),num_cpt,num_mic,num_sec,num_prop,num_azi,num_cf,vectorize_method = 3)
    cos_t_v_t_r          = vectorize(np.cos(t_v0+t_r0),num_cpt,num_mic,num_sec,num_prop,num_azi,num_cf,vectorize_method = 3)
    sin_t_v_t_r          = vectorize(np.sin(t_v0+t_r0),num_cpt,num_mic,num_sec,num_prop,num_azi,num_cf,vectorize_method = 3)
    M_hub                = vectorize(prop_origin,num_cpt,num_mic,num_sec,num_prop,num_azi,num_cf,vectorize_method = 5)
    POS_2                = vectorize(mls,num_cpt,num_mic,num_sec,num_prop,num_azi,num_cf,vectorize_method = 6)

    # ------------------------------------------------------------
    # ****** COORDINATE TRANSFOMRATIONS ******  
    M_t      = np.zeros((num_cpt,num_mic,num_prop,num_sec,num_azi,num_cf,3,3))
    M_phi    = np.zeros((num_cpt,num_mic,num_prop,num_sec,num_azi,num_cf,3,3))
    M_theta  = np.zeros((num_cpt,num_mic,num_prop,num_sec,num_azi,num_cf,3,3))
    M_tv     = np.zeros((num_cpt,num_mic,num_prop,num_sec,num_azi,num_cf,3,3))

    M_tv[:,:,:,:,:,:,0,0]    = cos_t_v[:,:,:,:,:,:,0]
    M_tv[:,:,:,:,:,:,0,2]    = sin_t_v[:,:,:,:,:,:,0]
    M_tv[:,:,:,:,:,:,1,1]    = 1
    M_tv[:,:,:,:,:,:,2,0]    =-sin_t_v[:,:,:,:,:,:,0]
    M_tv[:,:,:,:,:,:,2,2]    = cos_t_v[:,:,:,:,:,:,0]
    POS_1                    = np.matmul(M_tv,(POS_2 + M_hub)) # rotor hub position relative to center of aircraft

    # twist angle matrix
    M_theta[:,:,:,:,:,:,0,0] =  cos_alpha_eff[:,:,:,:,:,:,0]
    M_theta[:,:,:,:,:,:,0,2] =  sin_alpha_eff[:,:,:,:,:,:,0]
    M_theta[:,:,:,:,:,:,1,1] = 1
    M_theta[:,:,:,:,:,:,2,0] = -sin_alpha_eff[:,:,:,:,:,:,0]
    M_theta[:,:,:,:,:,:,2,2] =  cos_alpha_eff[:,:,:,:,:,:,0]

    # azimuth motion matrix
    M_phi[:,:,:,:,:,:,0,0] =  sin_phi[:,:,:,:,:,:,0]
    M_phi[:,:,:,:,:,:,0,1] = -cos_phi[:,:,:,:,:,:,0]
    M_phi[:,:,:,:,:,:,1,0] =  cos_phi[:,:,:,:,:,:,0]
    M_phi[:,:,:,:,:,:,1,1] =  sin_phi[:,:,:,:,:,:,0]
    M_phi[:,:,:,:,:,:,2,2] = 1

    # tilt motion matrix 
    M_t[:,:,:,:,:,:,0,0] =  cos_t_v_t_r[:,:,:,:,:,:,0]
    M_t[:,:,:,:,:,:,0,2] =  sin_t_v_t_r[:,:,:,:,:,:,0]
    M_t[:,:,:,:,:,:,1,1] =  1
    M_t[:,:,:,:,:,:,2,0] = -sin_t_v_t_r[:,:,:,:,:,:,0]
    M_t[:,:,:,:,:,:,2,2] =  cos_t_v_t_r[:,:,:,:,:,:,0] 
    
    # transformation of geographical global reference frame to the sectional local coordinate
    mat0    = np.matmul(M_t,POS_1)   
    mat1    = np.matmul(M_phi,mat0)
    POS     = np.matmul(M_theta,mat1)

    blade_section_position_vectors = Data()
    blade_section_position_vectors.blade_section_coordinate_sys    = POS 
    blade_section_position_vectors.vehicle_coordinate_sys          = POS_2
    blade_section_position_vectors.cos_phi                         = np.repeat(cos_phi,2,axis = 6)  
    blade_section_position_vectors.sin_alpha_eff                   = np.repeat(sin_alpha_eff,2,axis = 6)     
    blade_section_position_vectors.cos_alpha_eff                   = np.repeat(cos_alpha_eff,2,axis = 6) 
    blade_section_position_vectors.M_hub_X                         = np.repeat(M_hub[:,:,:,:,:,:,0,:],2,axis = 6)
    blade_section_position_vectors.M_hub_Y                         = np.repeat(M_hub[:,:,:,:,:,:,2,:],2,axis = 6)
    blade_section_position_vectors.M_hub_Z                         = np.repeat(M_hub[:,:,:,:,:,:,2,:],2,axis = 6)
    blade_section_position_vectors.cos_t_v                         = np.repeat(cos_t_v,2,axis = 6)
    blade_section_position_vectors.sin_t_v                         = np.repeat(sin_t_v,2,axis = 6)
    blade_section_position_vectors.cos_t_v_t_r                     = np.repeat(cos_t_v_t_r,2,axis = 6)
    blade_section_position_vectors.sin_t_v_t_r                     = np.repeat(sin_t_v_t_r,2,axis = 6)

    return blade_section_position_vectors

def vectorize(vec,num_cpt,num_mic,num_sec,num_prop,num_azi,num_cf,vectorize_method): 
    if vectorize_method == 1:
        
        # number of control points ,  number of microphones , number of rotors, rotor blade sections, number of azimuthal locations , broadband section resolution
        vec_x = np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(np.atleast_2d(vec),num_prop,axis = 0)[np.newaxis,:,:],\
                num_mic,axis = 0)[np.newaxis,:,:,:],num_cpt,axis = 0)[:,:,:,:,np.newaxis],num_azi,axis = 4)\
                [:,:,:,:,:,np.newaxis],num_cf,axis =5)
    elif vectorize_method == 2:
        vec_x = np.repeat(np.repeat(np.repeat(vec[:,np.newaxis,:,:],num_prop,axis = 1)[:,np.newaxis,:,:,:],\
                num_mic,axis = 1)[:,:,:,:,:,np.newaxis],num_cf,axis =5)[:,:,:,:,:,:,np.newaxis]
    
    elif vectorize_method == 3:
        vec_x = np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(vec[:,np.newaxis,:],num_prop,axis = 1)[:,np.newaxis,:,:],\
                num_mic,axis = 1)[:,:,:,np.newaxis],num_sec,axis = 3)[:,:,:,:,np.newaxis,:],num_azi,axis = 4)\
                [:,:,:,:,:,np.newaxis,:],num_cf,axis = 5)
        
    elif vectorize_method == 4:
        vec_x = np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(vec[:,np.newaxis,:],num_prop,axis = 1)\
                [:,np.newaxis,:,:], num_mic,axis = 1)[:,:,:,np.newaxis,:],num_sec,axis = 3),2,axis = 4)\
                [:,:,:,:,np.newaxis,:],num_azi,axis = 4)[:,:,:,:,:,np.newaxis,:],num_cf,axis = 5)
 
    elif vectorize_method == 5: 
        vec_x = np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(vec[np.newaxis,:,:],num_cpt,axis = 0)\
                [:,np.newaxis,:,:],num_mic, axis = 1)[:,:,:,np.newaxis,:],num_sec,axis = 3)[:,:,:,:,np.newaxis,:],\
                num_azi,axis = 4)[:,:,:,:,:,np.newaxis,:],num_cf,axis = 5)[:,:,:,:,:,:,:,np.newaxis]
 
    elif vectorize_method == 6: 
        vec_x = np.repeat(np.repeat(np.repeat(np.repeat(vec[:,:,np.newaxis,:],num_prop,axis = 2)\
                [:,:,:,np.newaxis,:],num_sec,axis = 3)[:,:,:,:,np.newaxis,:],num_azi,axis = 4)\
                [:,:,:,:,:,np.newaxis,:],num_cf,axis = 5)[:,:,:,:,:,:,:,np.newaxis]
  
    elif vectorize_method == 7: 
        vec_x = np.repeat(np.repeat(np.repeat(vec[np.newaxis,:,:],num_prop,axis = 0)[np.newaxis,:,:,:],\
                num_mic,axis = 0)[:,:,:,:,np.newaxis],num_cf,axis =4)
    
    elif vectorize_method == 8:  
        vec_x = np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(vec[np.newaxis,:],num_sec,axis=0)\
                [np.newaxis,:,:], num_prop,axis=0)[np.newaxis,:,:,:],num_mic,axis=0)[np.newaxis,:,:,:,:],num_cpt,axis=0)\
                [:,:,:,:,np.newaxis,:],num_azi,axis=4)[:,:,:,:,:,:,np.newaxis],2,axis=6) 

    elif vectorize_method == 9:  
        vec_x = np.repeat(vec,2,axis=6) 
 
    return vec_x