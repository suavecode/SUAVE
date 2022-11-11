## @ingroup Methods-Noise-Fidelity_One-Noise_Tools
# generate_microphone_points.py
# 
# Created: Sep 2021, M. Clarke  

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
from scipy.interpolate import griddata
import numpy as np

# ----------------------------------------------------------------------
#  Compute Microphone Points
# ---------------------------------------------------------------------
## @ingroup Methods-Noise-Fidelity_One-Noise_Tools 
def generate_ground_microphone_points(settings):
    """This computes the absolute microphone/observer locations on level ground. 
            
    Assumptions:
        None

    Source:
        N/A  

    Inputs:   
    settings.
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

    N_x                   = settings.ground_microphone_x_resolution 
    N_y                   = settings.ground_microphone_y_resolution
    num_gm                = N_x*N_y
    gm_mic_locations      = np.zeros((num_gm,3))     
    min_x                 = settings.ground_microphone_min_x         
    max_x                 = settings.ground_microphone_max_x         
    min_y                 = settings.ground_microphone_min_y         
    max_y                 = settings.ground_microphone_max_y   
    x_coords_0            = np.repeat(np.linspace(min_x,max_x,N_x)[:,np.newaxis],N_y, axis = 1)
    y_coords_0            = np.repeat(np.linspace(min_y,max_y,N_y)[:,np.newaxis],N_x, axis = 1).T
    z_coords_0            = np.zeros_like(x_coords_0) 
    gm_mic_locations[:,0] = x_coords_0.reshape(num_gm)
    gm_mic_locations[:,1] = y_coords_0.reshape(num_gm)
    gm_mic_locations[:,2] = z_coords_0.reshape(num_gm) 
    
    # store ground microphone locations
    settings.ground_microphone_locations = gm_mic_locations
    return   
  
## @ingroup Methods-Noise-Fidelity_One-Noise_Tools  
def preprocess_topography_and_route_data(topography_file,N_lat,N_long,departure_coordinates,destination_coordinates):
    """This computes the absolute microphone/observer locations on a defined topography
            
    Assumptions: 
        None

    Source:
        N/A  

    Inputs:  
        topography_file - file of lattide, longitude and elevation points  [-]
        N_lat           - discretization of points in latitude direction   [meters]
        N_long          - discretization of points in longitude direction  [meters] 
        departure_coordinates
        destination_coordinates
        
    Outputs: 
        cartesian_pts  - cartesian coordiates (x,y,z) of all microphones in domain                [meters]
        lat_long_pts   - latitude-longitude and elevation coordiates of all microphones in domain [deg,deg,m] 
        aircraft_range - 
        x0             -
        y0             -
        z0             -
        z_fin          -
        true_course    -
    
    
    Properties Used:
        N/A       
    """   
    data = np.loadtxt(topography_file)
    Long = data[:,0]
    Lat  = data[:,1]
    Elev = data[:,2]   

    earth                 = SUAVE.Attributes.Planets.Earth()
    R                     = earth.mean_radius      
    x_dist_max            = (np.max(Lat)-np.min(Lat))*Units.degrees * R  
    y_dist_max            = (np.max(Long)-np.min(Long))*Units.degrees * R   
    [long_dist,lat_dist]  = np.meshgrid(np.linspace(0,y_dist_max,N_long),np.linspace(0,x_dist_max,N_lat))
    [long_deg,lat_deg]    = np.meshgrid(np.linspace(np.min(Long),np.max(Long),N_long),np.linspace(np.min(Lat),np.max(Lat),N_lat)) 
    z_deg                 = griddata((Lat,Long), Elev, (lat_deg, long_deg), method='linear')       
    cartesian_pts         = np.dstack((np.dstack((lat_dist[:,:,None],long_dist[:,:,None] )),z_deg[:,:,None]))
    lat_long_pts          = np.dstack((np.dstack((lat_deg[:,:,None],long_deg[:,:,None] )),z_deg[:,:,None]))  
    cartesian_pts_vec     = cartesian_pts.reshape(N_lat*N_long,3)
    lat_long_pts_vec      = lat_long_pts.reshape(N_lat*N_long,3)
     
    # Compute coordinates of departure and destination points to determine range 
    coord0_rad            = departure_coordinates*Units.degrees
    coord1_rad            = destination_coordinates*Units.degrees  
    angle                 = np.arccos(np.sin(coord0_rad[0])*np.sin(coord1_rad[0]) + 
                                      np.cos(coord0_rad[0])*np.cos(coord1_rad[0])*np.cos(coord0_rad[1] - coord1_rad[1]))
    aircraft_range        = R*angle

    # Compute heading from departure to destination    
    true_course           = np.arcsin( np.sin(np.pi/2 - coord1_rad[0])* np.sin(coord1_rad[1] - coord0_rad[1])/np.sin(angle)) 
    angle_vector          = destination_coordinates - departure_coordinates 
    if angle_vector[0] < 0:
        true_course       = np.pi - true_course 
    
    # Compute relative location on topographical grid
    corner_lat            = lat_long_pts_vec[0,0]
    corner_long           = lat_long_pts_vec[0,1]
    if corner_long>180:
        corner_long       = corner_long-360  
    x0 = (coord0_rad[0] - corner_lat*Units.degrees)*R
    y0 = (coord0_rad[1] - corner_long*Units.degrees)*R  
   
    # Compute departure and destination elevation  
    flag_lat_locs  = np.where(departure_coordinates<0)[0]
    departure_coordinates[flag_lat_locs] = departure_coordinates[flag_lat_locs] + 360 
    flag_long_locs = np.where(destination_coordinates<0)[0]
    destination_coordinates[flag_long_locs] = destination_coordinates[flag_long_locs] + 360  

    z0      = griddata((Lat,Long), Elev, (np.array([departure_coordinates[0]]),np.array([departure_coordinates[1]])), method='linear')[0]
    z_fin   = griddata((Lat,Long), Elev, (np.array([destination_coordinates[0]]),np.array([destination_coordinates[1]])), method='linear')[0]
    
    return cartesian_pts_vec, lat_long_pts_vec,aircraft_range,x0,y0,z0,z_fin,true_course 
    
## @ingroup Methods-Noise-Fidelity_One-Noise_Tools 
def generate_building_microphone_points(building_locations,building_dimensions,N_x,N_y,N_z):
    """This computes the absolute microphone/observer locations on the surface of recti-linear buildings. 
            
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
        building_mic_locations         - cartesian coordiates of all microphones defined on buildings  [meters]
                              
    
    Properties Used:
        N/A       
    """   
    building_locations       = np.array(building_locations)
    building_dimensions      = np.array(building_dimensions)
    N_b                      = len(building_locations) 
    num_mics_on_xz_surface   = N_x*N_z 
    num_mics_on_yz_surface   = N_y*N_z 
    num_mics_on_xy_surface   = N_x*N_y
    num_mics_per_building    = 2*(num_mics_on_xz_surface +num_mics_on_yz_surface) +  num_mics_on_xy_surface 
    b_mic_locations          = np.empty((N_b,num_mics_per_building,3))

    x0 = building_locations[:,0] 
    y0 = building_locations[:,1] 
    z0 = building_locations[:,2] 
    l  = building_dimensions[:,0] 
    w  = building_dimensions[:,1] 
    h  = building_dimensions[:,2]     
    
    # surface 1 (front) 
    x_coords_1     = np.repeat(np.repeat(np.atleast_2d(x0-l/2).T,N_y,axis = 1)[:,:,np.newaxis],N_z,axis = 2)  
    Y_1            = np.repeat(np.repeat(np.atleast_2d(y0).T,N_y,axis = 1)[:,:,np.newaxis],N_z,axis = 2)
    YW_1           = np.repeat(np.repeat(np.atleast_2d(w/2).T,N_y,axis = 1)[:,:,np.newaxis],N_z,axis = 2)
    non_dim_y_1    = np.repeat(np.repeat(np.linspace(-1,1,N_y)[:,np.newaxis],N_z, axis = 1)[np.newaxis,:,:],N_b, axis = 0) 
    y_coords_1     = non_dim_y_1*YW_1 + Y_1 
    Z_1            = np.repeat(np.repeat(np.atleast_2d(h).T,N_y,axis = 1)[:,:,np.newaxis],N_z,axis = 2) 
    non_dim_z_1    = np.repeat(np.repeat(np.linspace(0,1,N_z)[:,np.newaxis],N_y, axis = 1).T[np.newaxis,:,:],N_b, axis = 0) 
    z_coords_1     = non_dim_z_1*Z_1  
    start_idx_1    = 0 
    end_idx_1      = num_mics_on_yz_surface  
    b_mic_locations[:,start_idx_1:end_idx_1 ,0] = x_coords_1.reshape(N_b,N_y*N_z)
    b_mic_locations[:,start_idx_1:end_idx_1 ,1] = y_coords_1.reshape(N_b,N_y*N_z)
    b_mic_locations[:,start_idx_1:end_idx_1 ,2] = z_coords_1.reshape(N_b,N_y*N_z) 
        
    # surface 2 (right)    
    X_2            = np.repeat(np.repeat(np.atleast_2d(x0).T,N_x,axis = 1)[:,:,np.newaxis],N_z,axis = 2)
    XW_2           = np.repeat(np.repeat(np.atleast_2d(l/2).T,N_x,axis = 1)[:,:,np.newaxis],N_z,axis = 2)
    non_dim_x_2    = np.repeat(np.repeat(np.linspace(-1,1,N_x)[:,np.newaxis],N_z, axis = 1)[np.newaxis,:,:],N_b, axis = 0) 
    x_coords_2     = non_dim_x_2*XW_2 + X_2   
    y_coords_2     = np.repeat(np.repeat(np.atleast_2d(y0+w/2).T,N_x,axis = 1)[:,:,np.newaxis],N_z,axis = 2)     
    Z_2            = np.repeat(np.repeat(np.atleast_2d(h).T,N_x,axis = 1)[:,:,np.newaxis],N_z,axis = 2) 
    non_dim_z_2    = np.repeat(np.repeat(np.linspace(0,1,N_z)[:,np.newaxis],N_x, axis = 1).T[np.newaxis,:,:],N_b, axis = 0) 
    z_coords_2     = non_dim_z_2*Z_2 
    start_idx_2    = end_idx_1    
    end_idx_2      = start_idx_2 + num_mics_on_xz_surface  
    b_mic_locations[:,start_idx_2:end_idx_2 ,0] = x_coords_2.reshape(N_b,N_y*N_z)
    b_mic_locations[:,start_idx_2:end_idx_2 ,1] = y_coords_2.reshape(N_b,N_y*N_z)
    b_mic_locations[:,start_idx_2:end_idx_2 ,2] = z_coords_2.reshape(N_b,N_y*N_z)  
        
    # surface 3 (back) 
    x_coords_3     = np.repeat(np.repeat(np.atleast_2d(x0+l/2).T,N_y,axis = 1)[:,:,np.newaxis],N_z,axis = 2) 
    Y_3            = np.repeat(np.repeat(np.atleast_2d(y0).T,N_y,axis = 1)[:,:,np.newaxis],N_z,axis = 2)
    YW_3           = np.repeat(np.repeat(np.atleast_2d(w/2).T,N_y,axis = 1)[:,:,np.newaxis],N_z,axis = 2)
    non_dim_y_3    = np.repeat(np.repeat(np.linspace(-1,1,N_y)[:,np.newaxis],N_z, axis = 1)[np.newaxis,:,:],N_b, axis = 0) 
    y_coords_3     = non_dim_y_3*YW_3 +  Y_3   
    Z_3            = np.repeat(np.repeat(np.atleast_2d(h).T,N_y,axis = 1)[:,:,np.newaxis],N_z,axis = 2) 
    non_dim_z_3    = np.repeat(np.repeat(np.linspace(0,1,N_z)[:,np.newaxis],N_y, axis = 1).T[np.newaxis,:,:],N_b, axis = 0) 
    z_coords_3     = non_dim_z_3*Z_3  
    start_idx_3    = end_idx_2 
    end_idx_3      = start_idx_3+num_mics_on_yz_surface  
    b_mic_locations[:,start_idx_3:end_idx_3 ,0] = x_coords_3.reshape(N_b,N_y*N_z)
    b_mic_locations[:,start_idx_3:end_idx_3 ,1] = y_coords_3.reshape(N_b,N_y*N_z)
    b_mic_locations[:,start_idx_3:end_idx_3 ,2] = z_coords_3.reshape(N_b,N_y*N_z)  
        
    # surface 4 (left)
    X_4            = np.repeat(np.repeat(np.atleast_2d(x0).T,N_x,axis = 1)[:,:,np.newaxis],N_z,axis = 2)      
    XW_4           = np.repeat(np.repeat(np.atleast_2d(l/2).T,N_x,axis = 1)[:,:,np.newaxis],N_z,axis = 2)      
    non_dim_x_4    = np.repeat(np.repeat(np.linspace(-1,1,N_x)[:,np.newaxis],N_z, axis = 1)[np.newaxis,:,:],N_b, axis = 0)       
    x_coords_4     = non_dim_x_4*XW_4 + X_4          
    y_coords_4     = np.repeat(np.repeat(np.atleast_2d(y0-w/2).T,N_x,axis = 1)[:,:,np.newaxis],N_z,axis = 2)          
    z_coords_4     = np.repeat(np.linspace(z0,h,(N_z))[:,np.newaxis],N_x, axis = 1).T        
    Z_4            = np.repeat(np.repeat(np.atleast_2d(h).T,N_x,axis = 1)[:,:,np.newaxis],N_z,axis = 2) 
    non_dim_z_4    = np.repeat(np.repeat(np.linspace(0,1,N_z)[:,np.newaxis],N_x, axis = 1).T[np.newaxis,:,:],N_b, axis = 0) 
    z_coords_4     = non_dim_z_4*Z_4 
    start_idx_4    = end_idx_3 
    end_idx_4      = start_idx_4 + num_mics_on_xz_surface      
    b_mic_locations[:,start_idx_4:end_idx_4 ,0] = x_coords_4.reshape(N_b,N_y*N_z)      
    b_mic_locations[:,start_idx_4:end_idx_4 ,1] = y_coords_4.reshape(N_b,N_y*N_z)      
    b_mic_locations[:,start_idx_4:end_idx_4 ,2] = z_coords_4.reshape(N_b,N_y*N_z)      
    
    # surface 5 (top) 
    X_5            = np.repeat(np.repeat(np.atleast_2d(x0).T,N_x,axis = 1)[:,:,np.newaxis],N_y,axis = 2)
    XW_5           = np.repeat(np.repeat(np.atleast_2d(l/2).T,N_x,axis = 1)[:,:,np.newaxis],N_y,axis = 2)
    non_dim_x_5    = np.repeat(np.repeat(np.linspace(-1,1,N_x)[:,np.newaxis],N_y, axis = 1)[np.newaxis,:,:],N_b, axis = 0) 
    x_coords_5     = non_dim_x_5*XW_5 + X_5    
    Y_5            = np.repeat(np.repeat(np.atleast_2d(y0).T,N_y,axis = 1)[:,np.newaxis,:],N_x,axis = 1)
    YW_5           = np.repeat(np.repeat(np.atleast_2d(w/2).T,N_y,axis = 1)[:,np.newaxis,:],N_x,axis = 1)
    non_dim_y_5    = np.repeat(np.repeat(np.linspace(-1,1,N_y)[:,np.newaxis],N_x, axis = 1).T[np.newaxis,:,:],N_b, axis = 0) 
    y_coords_5     = non_dim_y_5*YW_5 + Y_5     
    z_coords_5     = np.repeat(np.repeat(np.atleast_2d(h).T,N_x,axis = 1)[:,:,np.newaxis],N_y,axis = 2) 
    start_idx_5    = num_mics_on_xy_surface
    b_mic_locations[:,-start_idx_5:,0] = x_coords_5.reshape(N_b,N_x*N_y)
    b_mic_locations[:,-start_idx_5:,1] = y_coords_5.reshape(N_b,N_x*N_y)
    b_mic_locations[:,-start_idx_5:,2] = z_coords_5.reshape(N_b,N_x*N_y)    
    
    building_mic_locations =  b_mic_locations.reshape(N_b*num_mics_per_building,3)    
             
    return building_mic_locations
