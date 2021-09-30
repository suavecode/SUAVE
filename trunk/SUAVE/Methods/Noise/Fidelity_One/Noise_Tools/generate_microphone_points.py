## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# generate_microphone_points.py
# 
# Created: Sep 2021, M. Clarke  

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#  Compute Microphone Points
# ---------------------------------------------------------------------
## @ingroupMethods-Noise-Fidelity_One-Noise_Tools 
def generate_ground_microphone_points(min_x,max_x,min_y,max_y,N_x,N_y):
    """This computes the absolute microphone/observer locations on level ground. 
            
    Assumptions:
        None

    Source:
        N/A  

    Inputs:   
        min_x - minimum x coordinate of noise evaluation plane [meters]
        max_x - maximum x coordinate of noise evaluation plane [meters]
        min_y - minimum y coordinate of noise evaluation plane [meters]
        max_x - maximim y coordinate of noise evaluation plane [meters]
        N_x   - number of microphones on x-axis 
        N_y   - number of microphones on y-axis 
    
    Outputs: 
        gm_mic_locations   - cartesian coordiates of all microphones defined  [meters] 
    
    Properties Used:
        N/A       
    """       
    num_gm = N_x*N_y
    gm_mic_locations      = np.zeros((num_gm,3))    
    x_coords_0            = np.repeat(np.linspace(min_x,max_x,N_x)[:,np.newaxis],N_y, axis = 1)
    y_coords_0            = np.repeat(np.linspace(min_y,max_y,N_y)[:,np.newaxis],N_x, axis = 1).T
    z_coords_0            = np.zeros_like(x_coords_0) 
    gm_mic_locations[:,0] = x_coords_0.reshape(num_gm)
    gm_mic_locations[:,1] = y_coords_0.reshape(num_gm)
    gm_mic_locations[:,2] = z_coords_0.reshape(num_gm)     
    
    return gm_mic_locations 
     

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools 
def generate_building_microphone_points(building_locations,building_dimensions,N_x,N_y,N_z):
    """This computes the absolute microphone/observer locations on the surface of rectilinear buildinsg. 
            
    Assumptions:
        Microhpone locations are uniformly distributed on the surface

    Source:
        N/A  

    Inputs:  
        building_locations             - cartesian coordinates of the base of buildings                [meters]
        building_dimensions            - dimensions of buildings [length,width,height]                 [meters] 
        building_microphone_resolution - resolution of microphone array                                [unitless]
        N_x                            - discretization of points in x dimension on building surface   [meters]
        N_y                            - discretization of points in y dimension on building surface   [meters]
        N_z                            - discretization of points in z dimension on building surface   [meters]
        
    Outputs: 
        b_mic_locations                - cartesian coordiates of all microphones defined on buildings  [meters]
                              
    
    Properties Used:
        N/A       
    """   
    
    num_buildings            = len(building_locations) 
    num_mics_on_xz_surface   = N_x*N_z 
    num_mics_on_yz_surface   = N_y*N_z 
    num_mics_on_xy_surface   = N_x*N_y
    num_mics_per_building    = 2*(num_mics_on_xz_surface +num_mics_on_yz_surface) +  num_mics_on_xy_surface
    b_mic_locations          = np.empty((num_mics_per_building*num_buildings,3))
    
    for i in range(num_buildings): 
        x0 = building_locations[i][0] 
        y0 = building_locations[i][1] 
        z0 = building_locations[i][2] 
        l  = building_dimensions[i][0] 
        w  = building_dimensions[i][1] 
        h  = building_dimensions[i][2] 
        
        # define building microphones 
        building_microphones = np.zeros((num_mics_per_building,3)) # noise not computed on lower surface of buildings
        # surface 1 (front)
        x_coords_1  = np.ones((N_y,N_z))*(x0-l/2)
        y_coords_1  = np.repeat(np.linspace(y0-(w/2),y0+(w/2),N_y)[:,np.newaxis],N_z, axis = 1)
        z_coords_1  = np.repeat(np.linspace(z0,h,(N_z))[:,np.newaxis],N_y, axis = 1) .T
        start_idx_1 = 0 
        end_idx_1   = num_mics_on_yz_surface  
        building_microphones[start_idx_1:end_idx_1 ,0] = x_coords_1.reshape(N_y*N_z)
        building_microphones[start_idx_1:end_idx_1 ,1] = y_coords_1.reshape(N_y*N_z)
        building_microphones[start_idx_1:end_idx_1 ,2] = z_coords_1.reshape(N_y*N_z)
        
        # surface 2 (right)
        x_coords_2  = np.repeat(np.linspace(x0-(l/2),x0+(l/2),N_x)[:,np.newaxis],N_z)
        y_coords_2  = np.ones((N_x,N_z))*(y0+w/2)
        z_coords_2  = np.repeat(np.linspace(z0,h,(N_z))[:,np.newaxis],N_x, axis = 1).T
        start_idx_2 = end_idx_1
        end_idx_2   = start_idx_2 + num_mics_on_xz_surface 
        building_microphones[start_idx_2:end_idx_2,0] = x_coords_2.reshape(N_x*N_z)
        building_microphones[start_idx_2:end_idx_2,1] = y_coords_2.reshape(N_x*N_z) 
        building_microphones[start_idx_2:end_idx_2,2] = z_coords_2.reshape(N_x*N_z) 
        
        # surface 3 (back) 
        x_coords_3  = np.ones((N_y,N_z))*(x0+l/2)
        y_coords_3  = np.repeat(np.linspace(y0-(w/2),y0+(w/2),N_y)[:,np.newaxis],N_z, axis = 1)
        z_coords_3  = np.repeat(np.linspace(z0,h,(N_z))[:,np.newaxis],N_y, axis = 1) .T
        start_idx_3 = end_idx_2 
        end_idx_3   = start_idx_3+num_mics_on_yz_surface  
        building_microphones[start_idx_3:end_idx_3 ,0] = x_coords_3.reshape(N_y*N_z)
        building_microphones[start_idx_3:end_idx_3 ,1] = y_coords_3.reshape(N_y*N_z)
        building_microphones[start_idx_3:end_idx_3 ,2] = z_coords_3.reshape(N_y*N_z) 
         
        # surface 4 (left)
        x_coords_4  = np.repeat(np.linspace(x0-(l/2),x0+(l/2),N_x)[:,np.newaxis],N_z)
        y_coords_4  = np.ones((N_x,N_z))*(y0-w/2)
        z_coords_4  = np.repeat(np.linspace(z0,h,(N_z))[:,np.newaxis],N_x, axis = 1).T
        start_idx_4 = end_idx_3 
        end_idx_4   = start_idx_4 + num_mics_on_xz_surface 
        building_microphones[start_idx_4:end_idx_4,0] = x_coords_4.reshape(N_x*N_z)
        building_microphones[start_idx_4:end_idx_4,1] = y_coords_4.reshape(N_x*N_z) 
        building_microphones[start_idx_4:end_idx_4,2] = z_coords_4.reshape(N_x*N_z) 
       
        # surface 5 (top)
        x_coords_5 = np.repeat(np.linspace(x0-(l/2),x0+(l/2),N_x)[:,np.newaxis],N_y, axis = 1)
        y_coords_5 = np.repeat(np.linspace(y0-(w/2),y0+(w/2),N_y)[:,np.newaxis],N_x, axis = 1).T
        z_coords_5 = np.ones((N_x,N_y))*h     
        building_microphones[-num_mics_on_xy_surface:,0] = x_coords_5.reshape(N_x*N_y)
        building_microphones[-num_mics_on_xy_surface:,1] = y_coords_5.reshape(N_x*N_y)
        building_microphones[-num_mics_on_xy_surface:,2] = z_coords_5.reshape(N_x*N_y)    
        
        # append locations  
        start                        = i*num_mics_per_building
        end                          = (i+1)*num_mics_per_building 
        b_mic_locations[start:end,:] = building_microphones
        
    return b_mic_locations 
