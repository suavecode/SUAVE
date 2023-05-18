
## @ingroup Visualization-Geometry-Common
# post_process_noise_dat.py
#
# Created : Dec. 2022, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import numpy as np    
from MARC.Core import Data  
from scipy.interpolate import RegularGridInterpolator 
from MARC.Methods.Noise.Common.background_noise     import background_noise

## @ingroup Visualization-Geometry-Common
def post_process_noise_data(results,time_step = 20): 
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
    background_noise_dbA = background_noise()
    N_segs               = len(results.segments)
    N_ctrl_pts = 1
    for i in range(N_segs):  
        N_ctrl_pts  += len(results.segments[i].conditions.frames.inertial.time[:,0]) - 1  
    N_gm_x               = results.segments[0].analyses.noise.settings.ground_microphone_x_resolution
    N_gm_y               = results.segments[0].analyses.noise.settings.ground_microphone_y_resolution    
    SPL_dBA_t            = np.zeros((N_ctrl_pts ,N_gm_x,N_gm_y)) 
    time_old             = np.zeros(N_ctrl_pts)
    Aircraft_pos         = np.zeros((N_ctrl_pts,3)) 
    Mic_pos_gm           = results.segments[0].conditions.noise.total_ground_microphone_locations[0].reshape(N_gm_x,N_gm_y,3) 
     
    # Step 1: Merge data from all segments 
    idx = 0 
    for i in range(N_segs):  
        if  results.segments[i].battery_discharge == False:
            pass
        else:  
            if i == 0:
                start = 0 
            else:
                start = 1                  
            S_gm_x      = results.segments[i].analyses.noise.settings.ground_microphone_x_stencil
            S_gm_y      = results.segments[i].analyses.noise.settings.ground_microphone_y_stencil
            S_locs      = results.segments[i].conditions.noise.ground_microphone_stencil_locations
            x0          = results.segments[i].analyses.noise.settings.aircraft_departure_location[0]
            y0          = results.segments[i].analyses.noise.settings.aircraft_departure_location[1]
            N_ctrl_pts  = len(results.segments[i].conditions.frames.inertial.time[:,0])  
            for j in range(start,N_ctrl_pts): 
                time_old[idx]          = results.segments[i].conditions.frames.inertial.time[j,0]
                Aircraft_pos[idx,0]    = results.segments[i].conditions.frames.inertial.position_vector[j,0]  + x0
                Aircraft_pos[idx,1]    = results.segments[i].conditions.frames.inertial.position_vector[j,1]  + y0 
                x_idx                  = abs(Mic_pos_gm[:,0,0] - Aircraft_pos[idx,0]).argmin()
                y_idx                  = abs(Mic_pos_gm[0,:,1] - Aircraft_pos[idx,1]).argmin() 
                Aircraft_pos[idx,2]    = -results.segments[i].conditions.frames.inertial.position_vector[j,2] + Mic_pos_gm[x_idx,y_idx,2]
                stencil_length         = S_gm_x*2 + 1
                stencil_width          = S_gm_y*2 + 1
                SPL_dBA_t[idx,int(S_locs[j,0]):int(S_locs[j,1]),int(S_locs[j,2]):int(S_locs[j,3])]  = results.segments[i].conditions.noise.total_SPL_dBA[j].reshape(stencil_length ,stencil_width )  
                idx  += 1
                
    # Step 2: Make any readings less that background noise equal to background noise
    SPL_dBA                               = np.nan_to_num(SPL_dBA_t) 
    SPL_dBA[SPL_dBA<background_noise_dbA] = background_noise_dbA  
    
    # Step 3: Interpolate spacial and acoustic data based on time step 
    n_steps = int(np.floor(time_old[-1]/time_step))
    t_new   = np.linspace(0,n_steps*time_step,n_steps+ 1)  
    
    t_old_prime      = time_old
    t_old_prime[0]   = -0.01 # change the beginning and end point to allow 3-D interpolation 
    t_old_prime[-1]  = time_old[-1]+0.01 # change the beginning and end point to allow 3-D interpolation 
    
    # Noise Interpolation 
    t_1d         = np.tile(t_new[:,None,None],(1,N_gm_x,N_gm_y))
    x_1d         = np.tile(Mic_pos_gm[:,:,0][None,:,:],(len(t_new),1,1))
    y_1d         = np.tile(Mic_pos_gm[:,:,1][None,:,:],(len(t_new),1,1))  
    interp1      = RegularGridInterpolator((t_old_prime,Mic_pos_gm[:,0,0], Mic_pos_gm[0,:,1] ), SPL_dBA)  
    pts2         = np.concatenate((np.stack((t_1d.flatten(),x_1d.flatten()),axis = 1) ,y_1d.flatten()[:,None]),axis = 1)
    SPL_dBA_new  = interp1(pts2).reshape((len(t_new),N_gm_x,N_gm_y)) 
   
    # Temporal interpolation  
    Aircraft_pos_new      = np.zeros((n_steps+1,3))     
    Aircraft_pos_new[:,0] = np.interp(t_new,time_old, Aircraft_pos[:,0])
    Aircraft_pos_new[:,1] = np.interp(t_new,time_old, Aircraft_pos[:,1])
    Aircraft_pos_new[:,2] = np.interp(t_new,time_old, Aircraft_pos[:,2]) 
    
    # store data 
    noise_data                                  = Data()
    # noise  
    noise_data.SPL_dBA                          = SPL_dBA_new 
    
    # time stamp 
    noise_data.time                             = t_new 
    
    # microphone locations 
    noise_data.ground_microphone_locations      = Mic_pos_gm
    noise_data.ground_microphone_coordinates    = results.segments[0].analyses.noise.settings.ground_microphone_coordinates.reshape(N_gm_x,N_gm_y,3)  
    noise_data.ground_microphone_y_resolution   = N_gm_y
    noise_data.ground_microphone_x_resolution   = N_gm_x    
    
    # aircraft position 
    noise_data.aircraft_position                = Aircraft_pos_new  
    noise_data.topography_file                  = results.segments[0].analyses.noise.settings.topography_file              
    noise_data.aircraft_departure_location      = results.segments[0].analyses.noise.settings.aircraft_departure_location             
    noise_data.aircraft_destination_location    = results.segments[0].analyses.noise.settings.aircraft_destination_location         
    noise_data.aircraft_departure_coordinates   = results.segments[0].analyses.noise.settings.aircraft_departure_coordinates          
    noise_data.aircraft_destination_coordinates = results.segments[0].analyses.noise.settings.aircraft_destination_coordinates      

    return noise_data