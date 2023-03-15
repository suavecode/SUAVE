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
 
    mic_stencil_x  = settings.ground_microphone_x_stencil      
    mic_stencil_y  = settings.ground_microphone_y_stencil    
    N_gm_x         = settings.ground_microphone_x_resolution   
    N_gm_y         = settings.ground_microphone_y_resolution   
    gml            = settings.ground_microphone_locations 
    pos            = segment.state.conditions.frames.inertial.position_vector
    true_course    = segment.state.conditions.frames.planet.true_course_angle
    ctrl_pts       = len(pos)  
    TGML           = np.repeat(gml[np.newaxis,:,:],ctrl_pts,axis=0) 
    
    if (mic_stencil_x*2 + 1) > N_gm_x:
        print("Resetting microphone stenxil in x direction")
        mic_stencil_x = np.floor(N_gm_x/2 - 1)
    
    if (mic_stencil_y*2 + 1) > N_gm_y:
        print("Resetting microphone stenxil in y direction")
        mic_stencil_y = np.floor(N_gm_y/2 - 1)      
    
    # index location that is closest to the position of the aircraft 
    stencil_center_x_locs   = np.argmin(abs(np.tile((pos[:,0]+ settings.aircraft_departure_location[0])[:,None,None],(1,N_gm_x,N_gm_y)) -  np.tile(gml[:,0].reshape(N_gm_x,N_gm_y)[None,:,:],(ctrl_pts,1,1))),axis = 1)[:,0] 
    stencil_center_y_locs   = np.argmin(abs(np.tile((pos[:,1]+ settings.aircraft_departure_location[1])[:,None,None],(1,N_gm_x,N_gm_y)) -  np.tile(gml[:,1].reshape(N_gm_x,N_gm_y)[None,:,:],(ctrl_pts,1,1))),axis = 2)[:,0]
    
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
    EGML         = np.zeros((ctrl_pts,num_gm_mic,3))   
    REGML        = np.zeros_like(EGML)
    for cpt in range(ctrl_pts):
        surface         = TGML[cpt,:,:].reshape((N_gm_x,N_gm_y,3))
        stencil         = surface[start_x[cpt]:end_x[cpt],start_y[cpt]:end_y[cpt],:].reshape(num_gm_mic,3,1)  # extraction of points 
        
        EGML[cpt]       = stencil[:,:,0]
        
        relative_locations           = np.zeros((num_gm_mic,3,1))
        relative_locations[:,0,0]    = stencil[:,0,0] -  (pos[cpt,0] + settings.aircraft_departure_location[0])
        relative_locations[:,1,0]    = stencil[:,1,0] -  (pos[cpt,1] + settings.aircraft_departure_location[1])
        relative_locations[:,2,0]    = stencil[:,2,0] -  (pos[cpt,2]) 

        # apply rotation of matrix about z axis to orient grid to true course direction
        rotated_points   = np.matmul(np.tile(np.linalg.inv(true_course[cpt])[None,:,:],(num_gm_mic,1,1)),relative_locations)   
        REGML[cpt,:,:]   = rotated_points[:,:,0]  
        #REGML[cpt,:,:]   = relative_locations[:,:,0] 
     
    return REGML,EGML,TGML,num_gm_mic,mic_stencil
 