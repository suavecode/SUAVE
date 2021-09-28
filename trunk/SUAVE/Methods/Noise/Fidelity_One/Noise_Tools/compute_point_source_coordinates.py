## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# compute_point_source_coordinates.py
# 
# Created:  Mar 2021, M. Clarke 
# Modified: Jul 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#  Compute Point Source Coordinates 
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
    position_vector          = np.zeros((num_cpt,num_mic,num_prop,3))
    position_vector[:,:,:,0] =  mat_4[:,:,:,0,0]
    position_vector[:,:,:,1] =  mat_4[:,:,:,1,0]
    position_vector[:,:,:,2] =  mat_4[:,:,:,2,0]

    return position_vector    