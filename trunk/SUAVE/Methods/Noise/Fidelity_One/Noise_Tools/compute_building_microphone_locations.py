## @ingroupMethods-Noise-Fidelity_One-Noise_Tools
# compute_building_microphone_locations.py
# 
# Created: Sep 2021, M. Clarke  

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#  Compute Building Microphone Locations 
# ---------------------------------------------------------------------

## @ingroupMethods-Noise-Fidelity_One-Noise_Tools 
def compute_building_microphone_locations(building_locations,building_dimensions,building_microphone_resolution = 3):
    """This computes the microphone/observer locations on the surface of rectilinear buildins. 
            
    Assumptions:
        Microhpone locations are uniformly distributed on the surface

    Source:
        N/A  

    Inputs:  
        building_locations   - cartesian coordinates of the base of buildings   [meters]
        building_dimensions  - dimensions of buildings [length,width,height]    [meters] 
        building_microphone_resolution - resolution of microphone array         [unitless]
    Outputs: 
        mic_locations        - cartesian coordiates of all microphones defined  [meters]
                               on buildings 
    
    Properties Used:
        N/A       
    """   
    
    num_buildings            = len (building_locations)
    N                        = building_microphone_resolution
    num_mics_on_side_surface = N**3
    num_mics_on_top_surface  = N**2
    num_mics_per_building    = 4*num_mics_on_side_surface +  num_mics_on_top_surface
    mic_locations            = np.empty((num_mics_per_building*num_buildings,3))
    
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
        x_coords_1 = np.ones((N,N*N))*(x0-l/2)
        y_coords_1 = np.repeat(np.linspace(y0-(w/2),y0+(w/2),N)[:,np.newaxis],N*N, axis = 1)
        z_coords_1 = np.repeat(np.linspace(z0,h,(N*N))[:,np.newaxis],N, axis = 1) .T
        building_microphones[0:num_mics_on_side_surface ,0] = x_coords_1.reshape(N**3)
        building_microphones[0:num_mics_on_side_surface ,1] = y_coords_1.reshape(N**3)
        building_microphones[0:num_mics_on_side_surface ,2] = z_coords_1.reshape(N**3)
        
        # surface 2 (right)
        x_coords_2 = np.repeat(np.linspace(x0-(l/2),x0+(l/2),N)[:,np.newaxis],N*N)
        y_coords_2 = np.ones((N,N*N))*(y0+w/2)
        z_coords_2 = np.repeat(np.linspace(z0,h,(N*N))[:,np.newaxis],N, axis = 1).T
        building_microphones[num_mics_on_side_surface:2*num_mics_on_side_surface ,0] = x_coords_2.reshape(N**3)
        building_microphones[num_mics_on_side_surface:2*num_mics_on_side_surface ,1] = y_coords_2.reshape(N**3) 
        building_microphones[num_mics_on_side_surface:2*num_mics_on_side_surface ,2] = z_coords_2.reshape(N**3) 
        
        # surface 3 (back)
        x_coords_3 = np.ones((N,N*N))*(x0+l/2)
        y_coords_3 = np.repeat(np.linspace(y0-(w/2),y0+(w/2),N)[:,np.newaxis],N*N, axis = 1)
        z_coords_3 =  np.repeat(np.linspace(z0,h,(N*N))[:,np.newaxis],N, axis = 1).T
        building_microphones[2*num_mics_on_side_surface:3*num_mics_on_side_surface ,0] = x_coords_3.reshape(N**3)
        building_microphones[2*num_mics_on_side_surface:3*num_mics_on_side_surface ,1] = y_coords_3.reshape(N**3)
        building_microphones[2*num_mics_on_side_surface:3*num_mics_on_side_surface ,2] = z_coords_3.reshape(N**3) 
        
        # surface 4 (left)
        x_coords_4 = np.repeat(np.linspace(x0-(l/2),x0+(l/2),N)[:,np.newaxis],N*N, axis = 1)
        y_coords_4 = np.ones((N,N*N))*(y0-w/2)
        z_coords_4 = np.repeat(np.linspace(z0,h,(N*N))[:,np.newaxis],N, axis = 1).T 
        building_microphones[3*num_mics_on_side_surface:4*num_mics_on_side_surface ,0] = x_coords_4.reshape(N**3)
        building_microphones[3*num_mics_on_side_surface:4*num_mics_on_side_surface ,1] = y_coords_4.reshape(N**3)
        building_microphones[3*num_mics_on_side_surface:4*num_mics_on_side_surface ,2] = z_coords_4.reshape(N**3) 
         
        # surface 5 (top)
        x_coords_5 = np.repeat(np.linspace(x0-(l/2),x0+(l/2),N)[:,np.newaxis],N, axis = 1)
        y_coords_5 = np.repeat(np.linspace(y0-(w/2),y0+(w/2),N)[:,np.newaxis],N, axis = 1).T
        z_coords_5 = np.ones((N,N))*h     
        building_microphones[-num_mics_on_top_surface:,0] = x_coords_5.reshape(N**2)
        building_microphones[-num_mics_on_top_surface:,1] = y_coords_5.reshape(N**2)
        building_microphones[-num_mics_on_top_surface:,2] = z_coords_5.reshape(N**2)    
        
        # append locations  
        start = i*num_mics_per_building
        end   = (i+1)*num_mics_per_building 
        mic_locations[start:end,:] = building_microphones
    return mic_locations 
