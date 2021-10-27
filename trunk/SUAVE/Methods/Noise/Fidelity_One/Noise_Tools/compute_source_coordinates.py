## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# compute_source_coordinates.py
# 
# Created: Mar 2021, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#  Source Coordinates 
# ---------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools 
def compute_point_source_coordinates(AoA,thrust_angle,mls,prop_origin):  
    """This calculated the position vector from a point source to the observer 
            
    Assumptions:
        N/A

    Source:
        N/A  

    Inputs:  
    AoA                   - angle of attack           [rad]
    thrust_angle          - thrust angle              [rad]
    mls                   - microphone locations      [m]
    prop_origin           - propeller/rotor orgin     [m]

    Outputs: 
        position vector   - position vector of points [m]

    Properties Used:
        N/A       
    """  
    
    # aquire dimension of matrix
    num_cpt         = len(AoA)
    num_mic         = len(mls[0,:,0])
    num_prop        = len(prop_origin)
    prop_origin     = np.array(prop_origin) 
    
    # [control point, microphone , propeller , 2D geometry matrix ]
    # rotation of propeller about y axis by thurst angle (one extra dimension for translations)
    rotation_1            = np.zeros((num_cpt,num_mic,num_prop,4,4))
    rotation_1[:,:,:,0,0] = np.cos(thrust_angle)           
    rotation_1[:,:,:,0,2] = np.sin(thrust_angle)                 
    rotation_1[:,:,:,1,1] = 1
    rotation_1[:,:,:,2,0] = -np.sin(thrust_angle) 
    rotation_1[:,:,:,2,2] = np.cos(thrust_angle)      
    rotation_1[:,:,:,3,3] = 1     
    
    # translation to location on propeller
    I                        = np.atleast_3d(np.eye(4)).T
    translation_1            = np.repeat(np.repeat(np.repeat(I,num_prop, axis = 0)[np.newaxis,:,:,:],num_mic, axis = 0)[np.newaxis,:,:,:,:],num_cpt, axis = 0)
    translation_1[:,:,:,0,3] = np.repeat(np.repeat(np.atleast_2d(prop_origin[:,0]),num_mic, axis = 0)[np.newaxis,:,:],num_cpt, axis = 0)     
    translation_1[:,:,:,1,3] = np.repeat(np.repeat(np.atleast_2d(prop_origin[:,1]),num_mic, axis = 0)[np.newaxis,:,:],num_cpt, axis = 0)           
    translation_1[:,:,:,2,3] = np.repeat(np.repeat(np.atleast_2d(prop_origin[:,2]),num_mic, axis = 0)[np.newaxis,:,:],num_cpt, axis = 0) 
    
    # rotation of vehicle about y axis by AoA 
    rotation_2            = np.zeros((num_cpt,num_mic,num_prop,4,4))
    AoA_mat               = np.repeat(np.repeat(AoA,num_mic,axis = 1)[:,:,np.newaxis],num_prop,axis = 2)                
    rotation_2[:,:,:,0,0] = np.cos(AoA_mat)           
    rotation_2[:,:,:,0,2] = np.sin(AoA_mat)                 
    rotation_2[:,:,:,1,1] = 1
    rotation_2[:,:,:,2,0] = -np.sin(AoA_mat) 
    rotation_2[:,:,:,2,2] = np.cos(AoA_mat)     
    rotation_2[:,:,:,3,3] = 1 
    
    # translation of vehicle to air  
    translate_2            = np.repeat(np.repeat(np.repeat(I,num_prop, axis = 0)[np.newaxis,:,:,:],num_mic, axis = 0)[np.newaxis,:,:,:,:],num_cpt, axis = 0)       
    translate_2[:,:,:,0,3] = np.repeat(mls[:,:,0][:,:,np.newaxis],num_prop, axis = 2) 
    translate_2[:,:,:,1,3] = np.repeat(mls[:,:,1][:,:,np.newaxis],num_prop, axis = 2) 
    translate_2[:,:,:,2,3] = np.repeat(mls[:,:,2][:,:,np.newaxis],num_prop, axis = 2) 
    
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
def compute_blade_section_source_coordinates(AoA,thrust_angle,mls,prop_origin,prop):  
    """This calculated the position vector from a point source to the observer 
            
    Assumptions:
        N/A
 
    Source:
        N/A  
 
    Inputs:  
    AoA                   - angle of attack           [rad]
    thrust_angle          - thrust angle              [rad]
    mls                   - microphone locations      [m]
    prop_origin           - propeller/rotor orgin     [m]
 
    Outputs:  
 
    Properties Used:
        N/A       
    """  
    
    # aquire dimension of matrix
    num_cpt         = len(AoA)
    num_mic         = len(mls[0,:,0])
    num_prop        = len(prop.origin)
    prop_origin     = np.array(prop.origin)  
         
    AoA                = angle_of_attack         # vehicle angle of attack  
    thrust_angle       = prop.orientation_euler_angles[1]      
    vehicle_position   = np.array([[0,0,1]])
    prop_origin        = np.array([[0,0,0]])      
    vehicle_position   = position_vector          # x component of position vector of propeller to microphone  
    vehicle_position   = vehicle_position[:,:,np.newaxis]    
    beta               = prop.twist_distribution                           # twist distribution   
    pi                 = np.pi  
    N_r                = 1  
    r_0                = np.linspace(-0.5,0.5,num_sec+1) # coordinated of blade corners 
    r                  = (r_0[:-1] + r_0[1:])/2 # centerpoints where noise/forces are computed   
    theta_0            = np.array([[0]])  # collective pitch angle that varies wih rotor thrust
    theta              = np.array([[6.]])   # twist angle
    phi                = np.array([[0]])   # azimuth angle  
    t                  = np.array([[0]]) # tite angle   
    beta_p             = np.array([[0]])   # blade flaping angle 
    t_v                = np.array([[0]]) # negative body angle    vehicle tilt angle between the vehicle hub plane and the geographical ground 
    t_r                = np.array([[0]]) # prop.orientation_euler_angles # rotor tilt angle between the rotor hub plane and the vehicle hub plane  

    # Update dimensions for computation   
    r            = vectorize_1(r,N_r,ctrl_pts,BSR)  
    theta_0      = vectorize_2(theta_0,N_r,num_sec,BSR) 
    theta        = vectorize_2(theta,N_r,num_sec,BSR)  
    phi          = vectorize_2(phi,N_r,num_sec,BSR)  
    beta_p       = vectorize_2(beta_p,N_r,num_sec,BSR)  
    t            = vectorize_2(t,N_r,num_sec,BSR)   
    t_v          = vectorize_2(t_v,N_r,num_sec,BSR)  
    t_r          = vectorize_2(t_r,N_r,num_sec,BSR)      
    POS_2        = vectorize_2(vehicle_position,N_r,num_sec,BSR) 
    M_hub        = vectorize_4(prop_origin,ctrl_pts,num_sec,BSR)     
 

    # ------------------------------------------------------------
    # ****** COORDINATE TRANSFOMRATIONS ****** 
    M_beta_p = np.zeros((ctrl_pts,N_r,num_sec,BSR,3,1))
    M_t      = np.zeros((ctrl_pts,N_r,num_sec,BSR,3,3))
    M_phi    = np.zeros((ctrl_pts,N_r,num_sec,BSR,3,3))
    M_theta  = np.zeros((ctrl_pts,N_r,num_sec,BSR,3,3))
    M_tv     = np.zeros((ctrl_pts,N_r,num_sec,BSR,3,3))

    M_tv[:,:,:,:,0,0]    = np.cos(t_v[:,:,:,:,0])
    M_tv[:,:,:,:,0,2]    = np.sin(t_v[:,:,:,:,0])
    M_tv[:,:,:,:,1,1]    = 1
    M_tv[:,:,:,:,2,0]    =-np.sin(t_v[:,:,:,:,0])
    M_tv[:,:,:,:,2,2]    = np.cos(t_v[:,:,:,:,0]) 

    # rotor hub position relative to center of aircraft 
    POS_1   = np.matmul(M_tv,(POS_2 + M_hub))  

    # twist angle matrix
    M_theta[:,:,:,:,0,0] =  np.cos(theta_0[:,:,:,:,0] + theta[:,:,:,:,0])            
    M_theta[:,:,:,:,0,2] =  np.sin(theta_0[:,:,:,:,0] + theta[:,:,:,:,0])
    M_theta[:,:,:,:,1,1] = 1
    M_theta[:,:,:,:,2,0] = -np.sin(theta_0[:,:,:,:,0] + theta[:,:,:,:,0])
    M_theta[:,:,:,:,2,2] =  np.cos(theta_0[:,:,:,:,0] + theta[:,:,:,:,0])

    # azimuth motion matrix
    M_phi[:,:,:,:,0,0] =  np.sin(phi[:,:,:,:,0])                                               
    M_phi[:,:,:,:,0,1] = -np.cos(phi[:,:,:,:,0])  
    M_phi[:,:,:,:,1,0] =  np.cos(phi[:,:,:,:,0])  
    M_phi[:,:,:,:,1,1] =  np.sin(phi[:,:,:,:,0])     
    M_phi[:,:,:,:,2,2] = 1 

    # tilt motion matrix 
    M_t[:,:,:,:,0,0] =  np.cos(t[:,:,:,:,0])                                          
    M_t[:,:,:,:,0,2] =  np.sin(t[:,:,:,:,0]) 
    M_t[:,:,:,:,1,1] =  1 
    M_t[:,:,:,:,2,0] = -np.sin(t[:,:,:,:,0]) 
    M_t[:,:,:,:,2,2] =  np.cos(t[:,:,:,:,0]) 

    # flapping motion matrix
    M_beta_p[:,:,:,:,0,0]  = -r*np.sin(beta_p[:,:,:,:,0])*np.cos(phi[:,:,:,:,0])  
    M_beta_p[:,:,:,:,1,0]  = -r*np.sin(beta_p[:,:,:,:,0])*np.sin(phi[:,:,:,:,0])
    M_beta_p[:,:,:,:,2,0]  =  r*np.cos(beta_p[:,:,:,:,0])    

    # transformation of geographical global reference frame to the sectional local coordinate 
    mat0    = np.matmul(M_t,POS_1)  + M_beta_p                                      
    mat1    = np.matmul(M_phi,mat0)                                                
    POS     = np.matmul(M_theta,mat1)                                                

    X   = np.repeat(POS[:,:,:,:,:,:],2,axis = 4)    
    X_2 = np.repeat(POS_2[:,:,:,:,:,:],2,axis = 4) 
     
    return propeller_position_vector


def vectorize_1(vec,N_r,ctrl_pts,BSR):
    # control points ,  number rotors, number blades , broadband section resolution, 1
    vec_x = np.repeat(np.repeat(np.repeat(np.atleast_2d(vec),N_r,axis = 0)[np.newaxis,:,:],ctrl_pts,axis = 0)[:,:,:,np.newaxis],BSR,axis =3)   
    return vec_x

def vectorize_2(vec,N_r,num_sec,BSR):
    # control points ,  number rotors, number blades , num sections , broadband section resolution,1
    vec_x = np.repeat(np.repeat(np.repeat(vec[:,np.newaxis,:],N_r,axis = 1)[:,:,np.newaxis],num_sec,axis = 2)[:,:,:,np.newaxis,:],BSR,axis = 3) 
    return  vec_x 

def vectorize_3(vec,N_r,num_sec,BSR):
    # control points ,  number rotors, number blades , num sections , broadband section resolution, num_surfaces(2)
    vec_x = np.repeat(np.repeat(np.repeat(np.repeat(vec[:,np.newaxis,:],N_r,axis = 1)[:,:,np.newaxis],num_sec,axis = 2),2,axis = 3)[:,:,:,np.newaxis,:],BSR,axis = 3)
    return vec_x

def vectorize_4(vec,ctrl_pts,num_sec,BSR): 
    # control points ,  number rotors, number blades , num sections , broadband section resolution, coordinates(3) , 1
    vec_x = np.repeat(np.repeat(np.repeat(vec[np.newaxis,:,:],ctrl_pts,axis = 0)[:,:,np.newaxis,:],num_sec,axis = 2)[:,:,:,np.newaxis,:],BSR,axis = 3)[:,:,:,:,:,np.newaxis]
    return vec_x 

def vectorize_5(vec,N_r,BSR):
    # number rotors, number blades , num sections , broadband section resolution
    vec_x = np.repeat(np.repeat(vec[np.newaxis,:],N_r,axis = 0)[:,:,np.newaxis],BSR,axis = 2)      
    return vec_x

def vectorize_6(vec,ctrl_pts,N_r,num_sec):
    # number rotors, number blades , num sections , broadband section resolution
    vec_x = np.repeat(np.repeat(np.repeat(np.repeat(vec[np.newaxis,:],num_sec,axis=0)[np.newaxis,:,:],N_r,axis=0)[np.newaxis,:,:,:],ctrl_pts,axis=0)[:,:,:,:,np.newaxis],2,axis=4)
    return vec_x


def vectorize_7(vec,ctrl_pts,N_r,num_sec):

    vec_x = np.repeat(np.repeat(np.repeat(np.repeat(vec[np.newaxis,:],num_sec,axis=0)[np.newaxis,:,:],N_r,axis=0)[np.newaxis,:,:,:],ctrl_pts,axis=0)[:,:,:,:,np.newaxis],2,axis=4)

    return vec_x