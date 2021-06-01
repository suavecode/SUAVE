## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# compute_point_source_coordinates.py
# 
# Created: Mar 2021, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#  Compute Point Source Coordinates 
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
    position_vector          = np.zeros((num_cpt,num_mic,num_prop,3))
    position_vector[:,:,:,0] = mat_4[:,:,:,0,0]
    position_vector[:,:,:,1] = mat_4[:,:,:,1,0]
    position_vector[:,:,:,2] = mat_4[:,:,:,2,0]
     
    return position_vector
 