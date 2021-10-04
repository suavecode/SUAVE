## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# compute_noise_evaluation_locations.py
# 
# Created: Sep 2021, M. Clarke  

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#  Compute Noise Evaluation Points
# ---------------------------------------------------------------------
## @ingroupMethods-Noise-Fidelity_One-Noise_Tools 
def compute_ground_noise_evaluation_locations(settings,segment):
    """This computes the relative locations on the surface in the computational domain where the 
    propogated sound is computed.   
            
    Assumptions: 
        Acoustic scattering is not modeled

    Source:
        N/A  

    Inputs:  
        settings.ground_microphone_locations                - array of microphone locations on the ground  [meters] 
        segment.conditions.frames.inertial.position_vector  - position of aircraft                         [boolean]                                          

    Outputs: 
    GM_THETA   - angle measured from ground microphone in the x-z plane from microphone to aircraft 
    GM_PHI     - angle measured from ground microphone in the y-z plane from microphone to aircraft 
    GML        - ground microphone locations
    num_gm_mic - number of ground microphones
 
    Properties Used:
        N/A       
    """       
    
    gml            = settings.ground_microphone_locations 
    pos            = segment.state.conditions.frames.inertial.position_vector 
    ctrl_pts       = len(pos) 
    num_gm_mic     = len(gml)
    gml_3d         = np.repeat(gml[np.newaxis,:,:],ctrl_pts,axis=0)
    GML            = np.zeros_like(gml_3d) 
    Aircraft_x     = np.repeat(np.atleast_2d(pos[:,0] ).T,num_gm_mic , axis = 1)
    Aircraft_y     = np.repeat(np.atleast_2d(pos[:,1]).T,num_gm_mic , axis = 1)
    Aircraft_z     = np.repeat(np.atleast_2d(-pos[:,2]).T,num_gm_mic , axis = 1)
    GML[:,:,0]     = Aircraft_x - gml_3d[:,:,0] 
    GML[:,:,1]     = Aircraft_y - gml_3d[:,:,1]        
    GML[:,:,2]     = Aircraft_z - gml_3d[:,:,2]
    GM_THETA       = np.arctan(GML[:,:,2]/GML[:,:,0]) 
    GM_PHI         = np.arctan(GML[:,:,2]/GML[:,:,1])   
     
    return GM_THETA,GM_PHI,GML,num_gm_mic

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools 
def compute_building_noise_evaluation_locations(settings,segment):
    """This computes the relative locations on the surface in the computational domain where the 
    propogated sound is computed.   
            
    Assumptions: 
        Acoustic scattering is not modeled

    Source:
        N/A  

    Inputs:  
        settings.urban_canyon_microphone_locations          - array of microphone locations on a building(s)  [meters] 
        segment.conditions.frames.inertial.position_vector  - position of aircraft                            [boolean]                                     

    Outputs: 
    BM_THETA   - angle measured from building microphone in the x-z plane from microphone to aircraft 
    BM_PHI     - angle measured from building microphone in the y-z plane from microphone to aircraft 
    UCML       - building microphone locations
    num_b_mic  - number of building microphones
    
    Properties Used:
        N/A       
    """        
    ucml        = settings.urban_canyon_microphone_locations 
    pos         = segment.state.conditions.frames.inertial.position_vector 
    ctrl_pts    = len(pos)   
     
    if type(ucml) is np.ndarray: # urban canyon microphone locations
        num_b_mic      = len(ucml)
        ucml_3d        = np.repeat(ucml[np.newaxis,:,:],ctrl_pts,axis=0)
        UCML           = np.zeros_like(ucml_3d) 
        Aircraft_x     = np.repeat(np.atleast_2d(pos[:,0] ).T,num_b_mic , axis = 1)
        Aircraft_y     = np.repeat(np.atleast_2d(pos[:,1]).T,num_b_mic , axis = 1)
        Aircraft_z     = np.repeat(np.atleast_2d(-pos[:,2]).T,num_b_mic , axis = 1)
        UCML[:,:,0]    = Aircraft_x - ucml_3d[:,:,0] 
        UCML[:,:,1]    = Aircraft_y - ucml_3d[:,:,1]        
        UCML[:,:,2]    = Aircraft_z - ucml_3d[:,:,2]
        BM_THETA       = np.arctan(UCML[:,:,2]/UCML[:,:,0]) 
        BM_PHI         = np.arctan(UCML[:,:,2]/UCML[:,:,1])    
    else:
        UCML           = np.empty(shape=[ctrl_pts,0,3])
        BM_THETA       = np.empty(shape=[ctrl_pts,0]) 
        BM_PHI         = np.empty(shape=[ctrl_pts,0])   
        num_b_mic      = 0 
     
    return BM_THETA,BM_PHI,UCML,num_b_mic