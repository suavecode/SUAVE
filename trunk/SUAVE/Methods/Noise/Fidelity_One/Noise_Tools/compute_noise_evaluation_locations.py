## @ingroup Methods-Noise-Fidelity_One-Noise_Tools
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
## @ingroup Methods-Noise-Fidelity_One-Noise_Tools 
def compute_ground_noise_evaluation_locations(settings,segment):
    """This computes the relative locations on the surface in the computational domain where the 
    propogated sound is computed. Vectors point from observer/microphone to aircraft/source  
            
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
    REGML      - relative evaluation ground microphone locations
    EGML       - evaluation ground microphone locations
    TGML       - total ground microphone locations
    num_gm_mic - number of ground microphones
 
    Properties Used:
        N/A       
    """       
 
    mic_stencil_x  = settings.microphone_x_stencil      
    mic_stencil_y  = settings.microphone_y_stencil    
    N_gm_x         = settings.microphone_x_resolution   
    N_gm_y         = settings.microphone_y_resolution   
    gml            = settings.ground_microphone_locations 
    pos            = segment.state.conditions.frames.inertial.position_vector
    true_course    = segment.state.conditions.frames.planet.true_course_rotation
    ctrl_pts       = len(pos)  
    TGML           = np.repeat(gml[np.newaxis,:,:],ctrl_pts,axis=0) # (cpts,mics,3)
    
    if (mic_stencil_x*2 + 1) > N_gm_x:
        print("Resetting microphone stenxil in x direction")
        mic_stencil_x = np.floor(N_gm_x/2 - 1)
    
    if (mic_stencil_y*2 + 1) > N_gm_y:
        print("Resetting microphone stenxil in y direction")
        mic_stencil_y = np.floor(N_gm_y/2 - 1)     
        
        
    stencil_center_x_locs   = np.argmin(abs(np.tile(pos[:,0][:,None],(1,N_gm_x)) - np.tile( gml[:,0].reshape(N_gm_x,N_gm_y)[:,0][None,:],(ctrl_pts,1))),axis = 1) 
    stencil_center_y_locs   = np.argmin(abs(np.tile(pos[:,1][:,None],(1,N_gm_y)) - np.tile(gml[:,1].reshape(N_gm_x,N_gm_y)[0,:][None,:],(ctrl_pts,1))),axis = 1)
    
    # modify location of stencil center point if at edge 
    # top 
    locs_1 = np.where(stencil_center_x_locs >= (N_gm_x-mic_stencil_x))[0]
    stencil_center_x_locs[locs_1] = stencil_center_x_locs[locs_1] - ( mic_stencil_x - (N_gm_x - (stencil_center_x_locs[locs_1] + 1)))
    
    # right 
    locs_2 = np.where(stencil_center_y_locs >= (N_gm_y-mic_stencil_y))[0]
    stencil_center_y_locs[locs_2] = stencil_center_x_locs[locs_2] - ( mic_stencil_y - (N_gm_y - (stencil_center_y_locs[locs_2]+1)))     
 
    # bottom
    locs_3 = np.where(stencil_center_x_locs <  (mic_stencil_x))[0]
    stencil_center_x_locs[locs_3] = stencil_center_x_locs[locs_3] + ( mic_stencil_x - stencil_center_x_locs[locs_3])
    
    # left
    locs_4 = np.where(stencil_center_y_locs <  (mic_stencil_y))[0]
    stencil_center_y_locs[locs_4] = stencil_center_y_locs[locs_4] + ( mic_stencil_y - stencil_center_y_locs[locs_4])
     
    start_x = stencil_center_x_locs - mic_stencil_x
    start_y = stencil_center_y_locs - mic_stencil_y
    end_x   = stencil_center_x_locs + mic_stencil_x + 1
    end_y   = stencil_center_y_locs + mic_stencil_y + 1
    
    mic_stencil      = np.zeros((ctrl_pts,4))
    mic_stencil[:,0] = start_x 
    mic_stencil[:,1] = end_x   
    mic_stencil[:,2] = start_y 
    mic_stencil[:,3] = end_y   
    
    num_gm_mic  = (mic_stencil_x*2 + 1)*(mic_stencil_y*2 + 1)
    EGML         = np.zeros((ctrl_pts,num_gm_mic ,3))   
    for cpt in range(ctrl_pts):
        surface      = TGML[cpt,:,:].reshape((N_gm_x,N_gm_y,3))
        stencil      = surface[start_x[cpt]:end_x[cpt],start_y[cpt]:end_y[cpt],:].reshape(num_gm_mic,3)  # extraction of points 
        stencil[:,0] = stencil[:,0] - np.ones(num_gm_mic)*surface[stencil_center_x_locs[cpt],0,0]   # shifting to x == 0
        stencil[:,1] = stencil[:,1] - np.ones(num_gm_mic)*surface[0,stencil_center_y_locs[cpt],1]   # shifting to y == 0
        stencil      = np.matmul(stencil,true_course[cpt])                                          # apply rotation of matrix about z axis to orient grid to true course direction
        stencil[:,1] = stencil[:,1] + np.ones(num_gm_mic)*surface[0,stencil_center_y_locs[cpt],1]   # shifting to y == stencil_center_y_locs
        stencil[:,0] = stencil[:,0] + np.ones(num_gm_mic)*surface[stencil_center_x_locs[cpt],0,0]   # shifting to x == stencil_center_x_locs 
        EGML[cpt]    = stencil
          
    REGML          = np.zeros_like(EGML)
    Aircraft_x     = np.repeat(np.atleast_2d(pos[:,0] ).T,num_gm_mic , axis = 1)
    Aircraft_y     = np.repeat(np.atleast_2d(pos[:,1]).T,num_gm_mic , axis = 1)
    Aircraft_z     = np.repeat(np.atleast_2d(-pos[:,2]).T,num_gm_mic , axis = 1)
    REGML[:,:,0]   = Aircraft_x - EGML[:,:,0] 
    REGML[:,:,1]   = Aircraft_y - EGML[:,:,1]        
    REGML[:,:,2]   = Aircraft_z - EGML[:,:,2]
    GM_THETA       = np.zeros_like(REGML[:,:,2])
    GM_PHI         = np.zeros_like(REGML[:,:,2])
    GM_THETA       = np.arctan(REGML[:,:,2]/REGML[:,:,0]) 
    GM_PHI         = np.arctan(REGML[:,:,2]/REGML[:,:,1])   
     
    return GM_THETA,GM_PHI,REGML,EGML,TGML,num_gm_mic,mic_stencil

## @ingroup Methods-Noise-Fidelity_One-Noise_Tools 
def compute_building_noise_evaluation_locations(settings,segment):
    """This computes the relative locations on the surface in the computational domain where the 
    propogated sound is computed.   
            
    Assumptions: 
        Acoustic scattering is not modeled

    Source:
        N/A  

    Inputs:  
        settings.building_microphone_locations              - array of microphone locations on a building(s)  [meters] 
        segment.conditions.frames.inertial.position_vector  - position of aircraft                            [boolean]                                     

    Outputs: 
    BM_THETA   - angle measured from building microphone in the x-z plane from microphone to aircraft 
    BM_PHI     - angle measured from building microphone in the y-z plane from microphone to aircraft 
    UCML       - building microphone locations
    num_b_mic  - number of building microphones
    
    Properties Used:
        N/A       
    """        
    ucml        = settings.building_microphone_locations
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
        BM_THETA       = np.zeros_like(UCML[:,:,2])
        BM_PHI         = np.zeros_like(UCML[:,:,2]) 
        BM_THETA       = np.arctan(UCML[:,:,2]/UCML[:,:,0]) 
        BM_PHI         = np.arctan(UCML[:,:,2]/UCML[:,:,1])    
    else:
        UCML           = np.empty(shape=[ctrl_pts,0,3])
        BM_THETA       = np.empty(shape=[ctrl_pts,0]) 
        BM_PHI         = np.empty(shape=[ctrl_pts,0])   
        num_b_mic      = 0 
     
    return BM_THETA,BM_PHI,UCML,num_b_mic