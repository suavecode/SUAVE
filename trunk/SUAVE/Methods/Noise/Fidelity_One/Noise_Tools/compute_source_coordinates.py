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
def compute_point_source_coordinates(conditions,network,mls,source):  
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
    translate_2               = np.repeat(np.repeat(np.repeat(I,num_prop, axis = 0)[np.newaxis,:,:,:],num_mic, axis = 0)[np.newaxis,:,:,:,:],num_cpt, axis = 0)       
    translate_2[:,:,:,0,3]    = np.repeat(mls[:,:,0][:,:,np.newaxis],num_prop, axis = 2) 
    translate_2[:,:,:,1,3]    = np.repeat(mls[:,:,1][:,:,np.newaxis],num_prop, axis = 2) 
    translate_2[:,:,:,2,3]    = np.repeat(mls[:,:,2][:,:,np.newaxis],num_prop, axis = 2) 

    # identity transformation 
    I0    = np.atleast_3d(np.array([[0,0,0,1]]))
    mat_0 = np.repeat(np.repeat(np.repeat(I0,num_prop, axis = 0)[np.newaxis,:,:,:],num_mic, axis = 0)[np.newaxis,:,:,:,:],num_cpt, axis = 0)

    # execute operation  
    mat_1 = np.matmul(rotation_1,mat_0) 
    mat_2 = np.matmul(translation_1,mat_1)
    mat_3 = np.matmul(rotation_2,mat_2) 
    mat_4 = np.matmul(translate_2,mat_3)
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
    precision       = settings.floating_point_precision
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
    BSR           = settings.broadband_spectrum_resolution # broadband spectrum resolution   
    r             = prop.radius_distribution   
    num_sec       = len(r)  
    phi_2d        = acoustic_outputs.disc_azimuthal_distribution   
    beta_p        = np.zeros_like(phi_2d)
    alpha_eff     = acoustic_outputs.disc_effective_angle_of_attack
    num_azi       = len(phi_2d[0,0,:]) 
    t_v           = -AoA                                        # vehicle tilt angle between the vehicle hub plane and the geographical ground 
    t_r           = np.ones_like(AoA)*(np.pi/2 - thrust_angle)  # rotor tilt angle between the rotor hub plane and the vehicle hub plane 

    # Update dimensions for computation   
    r          = vectorize_1(r,num_cpt,num_mic,num_prop,num_azi,BSR)   
    beta_p     = vectorize_2(beta_p,num_mic,num_prop,BSR)  
    phi        = vectorize_2(phi_2d,num_mic,num_prop,BSR)   
    alpha_eff  = vectorize_2(alpha_eff,num_mic,num_prop,BSR)      
    t_v        = vectorize_3(t_v,num_mic,num_prop,num_sec,num_azi,BSR)  
    t_r        = vectorize_3(t_r,num_mic,num_prop,num_sec,num_azi,BSR)    
    M_hub      = vectorize_5(prop_origin,num_cpt,num_mic,num_sec,num_azi,BSR)   
    POS_2      = vectorize_6(mls,num_prop,num_sec,num_azi,BSR)  
    
    r          = np.array(r        , dtype=precision)
    beta_p     = np.array(beta_p   , dtype=precision)
    phi        = np.array(phi      , dtype=precision)
    alpha_eff  = np.array(alpha_eff, dtype=precision)
    t_v        = np.array(t_v      , dtype=precision)
    t_r        = np.array(t_r      , dtype=precision)
    M_hub      = np.array(M_hub    , dtype=precision)
    POS_2      = np.array(POS_2    , dtype=precision)
    

    # ------------------------------------------------------------
    # ****** COORDINATE TRANSFOMRATIONS ****** 
    M_beta_p = np.zeros((num_cpt,num_mic,num_prop,num_sec,num_azi,BSR,3,1), dtype=precision)
    M_t      = np.zeros((num_cpt,num_mic,num_prop,num_sec,num_azi,BSR,3,3), dtype=precision)
    M_phi    = np.zeros((num_cpt,num_mic,num_prop,num_sec,num_azi,BSR,3,3), dtype=precision)
    M_theta  = np.zeros((num_cpt,num_mic,num_prop,num_sec,num_azi,BSR,3,3), dtype=precision)
    M_tv     = np.zeros((num_cpt,num_mic,num_prop,num_sec,num_azi,BSR,3,3), dtype=precision)

    M_tv[:,:,:,:,:,:,0,0]    = np.cos(t_v[:,:,:,:,:,:,0])
    M_tv[:,:,:,:,:,:,0,2]    = np.sin(t_v[:,:,:,:,:,:,0])
    M_tv[:,:,:,:,:,:,1,1]    = 1  
    M_tv[:,:,:,:,:,:,2,0]    =-np.sin(t_v[:,:,:,:,:,:,0])
    M_tv[:,:,:,:,:,:,2,2]    = np.cos(t_v[:,:,:,:,:,:,0]) 

    POS_1   = np.matmul(M_tv,(POS_2 + M_hub)) # rotor hub position relative to center of aircraft 

    # twist angle matrix
    M_theta[:,:,:,:,:,:,0,0] =  np.cos(alpha_eff[:,:,:,:,:,:,0])                                   
    M_theta[:,:,:,:,:,:,0,2] =  np.sin(alpha_eff[:,:,:,:,:,:,0])
    M_theta[:,:,:,:,:,:,1,1] = 1   
    M_theta[:,:,:,:,:,:,2,0] = -np.sin(alpha_eff[:,:,:,:,:,:,0])
    M_theta[:,:,:,:,:,:,2,2] =  np.cos(alpha_eff[:,:,:,:,:,:,0])

    # azimuth motion matrix
    M_phi[:,:,:,:,:,:,0,0] =  np.sin(phi[:,:,:,:,:,:,0])                                            
    M_phi[:,:,:,:,:,:,0,1] = -np.cos(phi[:,:,:,:,:,:,0])   
    M_phi[:,:,:,:,:,:,1,0] =  np.cos(phi[:,:,:,:,:,:,0])  
    M_phi[:,:,:,:,:,:,1,1] =  np.sin(phi[:,:,:,:,:,:,0])     
    M_phi[:,:,:,:,:,:,2,2] = 1 

    # tilt motion matrix 
    M_t[:,:,:,:,:,:,0,0] =  np.cos(t_v[:,:,:,:,:,:,0] + t_r[:,:,:,:,:,:,0] )                                           
    M_t[:,:,:,:,:,:,0,2] =  np.sin(t_v[:,:,:,:,:,:,0] + t_r[:,:,:,:,:,:,0] ) 
    M_t[:,:,:,:,:,:,1,1] =  1                      
    M_t[:,:,:,:,:,:,2,0] = -np.sin(t_v[:,:,:,:,:,:,0] + t_r[:,:,:,:,:,:,0] ) 
    M_t[:,:,:,:,:,:,2,2] =  np.cos(t_v[:,:,:,:,:,:,0] + t_r[:,:,:,:,:,:,0] ) 

    # flapping motion matrix
    M_beta_p[:,:,:,:,:,:,0,0]  = -r*np.sin(beta_p[:,:,:,:,:,:,0])*np.cos(phi[:,:,:,:,:,:,0])  
    M_beta_p[:,:,:,:,:,:,1,0]  = -r*np.sin(beta_p[:,:,:,:,:,:,0])*np.sin(phi[:,:,:,:,:,:,0]) 
    M_beta_p[:,:,:,:,:,:,2,0]  =  r*np.cos(beta_p[:,:,:,:,:,:,0])   
    
    # transformation of geographical global reference frame to the sectional local coordinate 
    mat0    = np.matmul(M_t,POS_1)  + M_beta_p                                         
    mat1    = np.matmul(M_phi,mat0)                                                    
    POS     = np.matmul(M_theta,mat1)                   
    
    blade_section_position_vectors = Data()
    blade_section_position_vectors.blade_section_coordinate_sys    = POS 
    blade_section_position_vectors.rotor_coordinate_sys            = POS_1
    blade_section_position_vectors.vehicle_coordinate_sys          = POS_2 
    blade_section_position_vectors.r                               = r        
    blade_section_position_vectors.beta_p                          = beta_p   
    blade_section_position_vectors.phi                             = phi      
    blade_section_position_vectors.alpha_eff                       = alpha_eff   
    blade_section_position_vectors.t_v                             = t_v      
    blade_section_position_vectors.t_r                             = t_r       
    blade_section_position_vectors.M_hub                           = M_hub    
 
    return blade_section_position_vectors


def vectorize_1(vec,num_cpt,num_mic,num_prop,num_azi,BSR):
    # number of control points ,  number of microphones , number of rotors, rotor blade sections, number of azimuthal locations , broadband section resolution
    vec_x = np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(np.atleast_2d(vec),num_prop,axis = 0)[np.newaxis,:,:],\
            num_mic,axis = 0)[np.newaxis,:,:,:],num_cpt,axis = 0)[:,:,:,:,np.newaxis],num_azi,axis = 4)\
            [:,:,:,:,:,np.newaxis],BSR,axis =5)   
    return vec_x 

def vectorize_2(vec,num_mic,num_prop,BSR): 
    vec_x = np.repeat(np.repeat(np.repeat(vec[:,np.newaxis,:,:],num_prop,axis = 1)[:,np.newaxis,:,:,:],\
            num_mic,axis = 1)[:,:,:,:,:,np.newaxis],BSR,axis =5)[:,:,:,:,:,:,np.newaxis]
    return vec_x

def vectorize_3(vec,num_mic,num_prop,num_sec,num_azi,BSR): 
    vec_x = np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(vec[:,np.newaxis,:],num_prop,axis = 1)[:,np.newaxis,:,:],\
            num_mic,axis = 1)[:,:,:,np.newaxis],num_sec,axis = 3)[:,:,:,:,np.newaxis,:],num_azi,axis = 4)\
            [:,:,:,:,:,np.newaxis,:],BSR,axis = 5) 
    return  vec_x 

def vectorize_4(vec,num_mic,num_prop,num_sec,num_azi,BSR): 
    vec_x = np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(vec[:,np.newaxis,:],num_prop,axis = 1)\
            [:,np.newaxis,:,:], num_mic,axis = 1)[:,:,:,np.newaxis,:],num_sec,axis = 3),2,axis = 4)\
            [:,:,:,:,np.newaxis,:],num_azi,axis = 4)[:,:,:,:,:,np.newaxis,:],BSR,axis = 5)
    return vec_x

def vectorize_5(vec,num_cpt,num_mic,num_sec,num_azi,BSR):  
    vec_x = np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(vec[np.newaxis,:,:],num_cpt,axis = 0)\
            [:,np.newaxis,:,:],num_mic, axis = 1)[:,:,:,np.newaxis,:],num_sec,axis = 3)[:,:,:,:,np.newaxis,:],\
            num_azi,axis = 4)[:,:,:,:,:,np.newaxis,:],BSR,axis = 5)[:,:,:,:,:,:,:,np.newaxis]
    return vec_x 

def vectorize_6(vec,num_prop,num_sec,num_azi,BSR): 
    vec_x = np.repeat(np.repeat(np.repeat(np.repeat(vec[:,:,np.newaxis,:],num_prop,axis = 2)\
            [:,:,:,np.newaxis,:],num_sec,axis = 3)[:,:,:,:,np.newaxis,:],num_azi,axis = 4)\
            [:,:,:,:,:,np.newaxis,:],BSR,axis = 5)[:,:,:,:,:,:,:,np.newaxis]
    return vec_x  

def vectorize_8(vec,num_mic,num_cpt,num_prop,num_azi,num_sec): 
    vec_x = np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(np.repeat(vec[np.newaxis,:],num_sec,axis=0)\
            [np.newaxis,:,:], num_prop,axis=0)[np.newaxis,:,:,:],num_mic,axis=0)[np.newaxis,:,:,:,:],num_cpt,axis=0)\
            [:,:,:,:,np.newaxis,:],num_azi,axis=4)[:,:,:,:,:,:,np.newaxis],2,axis=6)
    return vec_x