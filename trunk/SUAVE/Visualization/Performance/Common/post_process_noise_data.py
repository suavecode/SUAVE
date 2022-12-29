
## @ingroup Visualization-Geometry-Common
# post_process_noise_dat.py
#
# Created : Dec. 2022, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import numpy as np    
from SUAVE.Core import Data 

## @ingroup Visualization-Geometry-Common
def post_process_noise_data(results): 
    """This translates all noise data into metadata for plotting 
    
    Assumptions:
    None
    
    Source: 
 
    Inputs: results 
         
    Outputs: noise_data
    
    Properties Used:
    N/A
    """

    # unpack 
    background_noise_dbA = 35 
    N_segs               = len(results.segments)
    N_ctrl_pts = 0
    for i in range(N_segs):  
        N_ctrl_pts  += len(results.segments[0].conditions.frames.inertial.time[:,0])   
    N_gm_x               = results.segments[0].analyses.noise.settings.ground_microphone_x_resolution
    N_gm_y               = results.segments[0].analyses.noise.settings.ground_microphone_y_resolution   
    dim_mat              = N_segs*N_ctrl_pts 
    SPL_contour_gm       = np.ones((dim_mat,N_gm_x,N_gm_y))*background_noise_dbA 
    Aircraft_pos         = np.zeros((dim_mat,3)) 
    Mic_pos_gm           = results.segments[0].conditions.noise.total_ground_microphone_locations[0].reshape(N_gm_x,N_gm_y,3) 

    idx = 0 
    for i in range(N_segs):  
        if  results.segments[i].battery_discharge == False:
            pass
        else:      
            S_gm_x = results.segments[i].analyses.noise.settings.ground_microphone_x_stencil
            S_gm_y = results.segments[i].analyses.noise.settings.ground_microphone_y_stencil
            S_locs = results.segments[i].conditions.noise.ground_microphone_stencil_locations
            x0     = results.segments[i].analyses.noise.settings.aircraft_departure_location[0]
            y0     = results.segments[i].analyses.noise.settings.aircraft_departure_location[1]
            N_ctrl_pts  = len(results.segments[i].conditions.frames.inertial.time[:,0])  
            for j in range(N_ctrl_pts): 
                Aircraft_pos[idx,0]    = results.segments[i].conditions.frames.inertial.position_vector[j,0]  + x0
                Aircraft_pos[idx,1]    = results.segments[i].conditions.frames.inertial.position_vector[j,1]  + y0
                Aircraft_pos[idx,2]    = -results.segments[i].conditions.frames.inertial.position_vector[j,2] 
                stencil_length         = S_gm_x*2 + 1
                stencil_width          = S_gm_y*2 + 1
                SPL_contour_gm[idx,int(S_locs[j,0]):int(S_locs[j,1]),int(S_locs[j,2]):int(S_locs[j,3])]  = results.segments[i].conditions.noise.total_SPL_dBA[j].reshape(stencil_length ,stencil_width )   
                idx  += 1
    noise_data                        = Data()
    noise_data.SPL_dBA_ground_mic     = np.nan_to_num(SPL_contour_gm) 
    noise_data.aircraft_position      = Aircraft_pos
    noise_data.SPL_dBA_ground_mic_loc = Mic_pos_gm 
    noise_data.N_gm_y                 = N_gm_y
    noise_data.N_gm_x                 = N_gm_x  

    return noise_data
    
