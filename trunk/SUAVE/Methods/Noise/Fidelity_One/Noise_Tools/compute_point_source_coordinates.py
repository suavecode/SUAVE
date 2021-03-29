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
def compute_point_source_coordinates(i,mic_loc,p_idx,AoA,thrust_angle,mls,prop_origin):  
    """This calculated the position vector from a point source to the observer 
            
    Assumptions:
        N/A

    Source:
        N/A 

    Inputs:
    i                     - control point
    mic_loc               - index of microphone 
    p_idx                 - index of propeller or rotor 
    AoA                   - angle of attack
    thrust_angle          - thrust angle
    mls                   - microphone locations 
    prop_origin           - propeller/rotor orgin

    Outputs: 
        position vector   - position vector of points 

    Properties Used:
        N/A         
    """ 
    
    # rotation of propeller about y axis by thurst angle (one extra dimension for translations)
    rotation_1      = np.zeros((4,4))
    rotation_1[0,0] = np.cos(thrust_angle)           
    rotation_1[0,2] = np.sin(thrust_angle)                 
    rotation_1[1,1] = 1
    rotation_1[2,0] = -np.sin(thrust_angle) 
    rotation_1[2,2] = np.cos(thrust_angle)      
    rotation_1[3,3] = 1     
    
    # translation to location on propeller
    translation_1      = np.eye(4)
    translation_1[0,3] = prop_origin[p_idx][0]     
    translation_1[1,3] = prop_origin[p_idx][1]           
    translation_1[2,3] = prop_origin[p_idx][2] 
    
    # rotation of vehicle about y axis by AoA 
    rotation_2      = np.zeros((4,4))
    rotation_2[0,0] = np.cos(AoA)           
    rotation_2[0,2] = np.sin(AoA)                 
    rotation_2[1,1] = 1
    rotation_2[2,0] = -np.sin(AoA) 
    rotation_2[2,2] = np.cos(AoA)     
    rotation_2[3,3] = 1 
    
    # translation of vehicle to air 
    translate_2      = np.eye(4)
    translate_2[0,3] = mls[i,mic_loc,0]  
    translate_2[1,3] = mls[i,mic_loc,1]   
    translate_2[2,3] = mls[i,mic_loc,2] 
    
    mat_0  = np.array([[0],[0],[0],[1]])
    
    # execute operation  
    mat_1 = np.matmul(rotation_1,mat_0) 
    mat_2 = np.matmul(translation_1,mat_1)
    mat_3 = np.matmul(rotation_2,mat_2) 
    mat_4 = np.matmul(translate_2,mat_3)
    mat_4 = -mat_4
    
    x = mat_4[0,0] 
    y = mat_4[1,0] 
    z = mat_4[2,0] 
    
    position_vector = np.array([x,y,z])
    
    return position_vector