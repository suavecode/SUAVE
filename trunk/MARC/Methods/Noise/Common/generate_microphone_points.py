## @ingroup Methods-Noise-Common
# generate_microphone_points.py
# 
# Created: Sep 2021, M. Clarke  

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
import MARC
from MARC.Core import Units, Data
from scipy.interpolate import griddata
import numpy as np
from geopy.distance import geodesic as GD

# ----------------------------------------------------------------------
#  Compute Microphone Points
# ---------------------------------------------------------------------
## @ingroup Methods-Noise-Common 
def generate_zero_elevation_microphone_points(settings):
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

## @ingroup Methods-Noise-Common  
def generate_terrain_elevated_microphone_points(topography_file                = None,
                                         departure_coordinates                 = [0.0,0.0],
                                         destination_coordinates               = [0.0,0.0],
                                         ground_microphone_x_resolution        = 101,
                                         ground_microphone_y_resolution        = 101,
                                         ground_microphone_x_stencil           = 3,
                                         ground_microphone_y_stencil           = 3, 
                                         adjusted_cruise_distance                      = 0):
    """This computes the absolute microphone/observer locations on a defined topography
            
    Assumptions: 
        topography_file is a text file obtained from https://topex.ucsd.edu/cgi-bin/get_data.cgi
    
    Source:
        N/A  

    Inputs:  
        topography_file                        - file of lattide, longitude and elevation points                                [-]
        departure_coordinates                  - coordinates of departure location                                              [degrees]
        destination_coordinates                - coordinates of destimation location                                            [degrees]
        ground_microphone_x_resolution         - number of points on computational domain in latitudal direction                [-]
        ground_microphone_y_resolution         - number of points on computational domain in  longitidinal direction            [-] 
        ground_microphone_x_stencil            - number of points in stencil in latitudal direction                             [-] 
        adjusted_cruise_distance                       - distance used to modify cruise to ensure desired range is met                  [-]
        ground_microphone_y_stencil            - number of points in stencil in in longitidinal direction                       [-] 
        
    Outputs: 
    topography_data.
        ground_microphone_x_resolution         - number of points on computational domain in latitudal direction                [-] 
        ground_microphone_y_resolution         - number of points on computational domain in  longitidinal direction            [-]
        ground_microphone_x_stencil            - number of points in stencil in latitudal direction                             [-]
        ground_microphone_y_stencil            - number of points in stencil in in longitidinal direction                       [-]       
        ground_microphone_min_x                - x-location of start of computation domain                                      [meters]          
        ground_microphone_max_x                - x-location of end of computation domain                                        [meters]  
        ground_microphone_min_y                - y-location of start of computation domain                                      [meters]                
        ground_microphone_max_y                - y-location of end of computation domain                                        [meters]  
        cartesian_micrphone_locations          - cartesian coordinates (x,y,z) of all microphones in domain                     [meters]       
        latitude_longitude_micrphone_locations - latitude-longitude and elevation coordinates of all microphones in domain      [deg,deg,m]   
    
    Properties Used:
        N/A       
    """     
    # convert cooordinates to array 
    departure_coordinates   = np.asarray(departure_coordinates)
    destination_coordinates = np.asarray(destination_coordinates)
    
    # extract data from file 
    data  = np.loadtxt(topography_file)
    Long  = data[:,0]
    Lat   = data[:,1]
    Elev  = data[:,2] 
    
    x_min_coord = np.min(Lat)
    x_max_coord = np.max(Lat)
    y_min_coord = np.min(Long)
    y_max_coord = np.max(Long)
    if np.min(Long)>180: 
        y_min_coord = np.min(Long)-360
    if np.max(Long)>180:
        y_max_coord = np.max(Long)-360  
    
    top_left_map_coords      = np.array([x_max_coord,y_min_coord])
    bottom_left_map_coords   = np.array([x_min_coord,y_min_coord])  
    bottom_right_map_coords  = np.array([x_min_coord,y_max_coord]) 
    
    x_dist_max = GD(top_left_map_coords,bottom_left_map_coords).m 
    y_dist_max = GD(bottom_right_map_coords,bottom_left_map_coords).m  
    
    [y_pts,x_pts]      = np.meshgrid(np.linspace(0,y_dist_max,ground_microphone_y_resolution),np.linspace(0,x_dist_max,ground_microphone_x_resolution))
    [long_deg,lat_deg] = np.meshgrid(np.linspace(np.min(Long),np.max(Long),ground_microphone_y_resolution),np.linspace(np.min(Lat),np.max(Lat),ground_microphone_x_resolution)) 
    z_deg              = griddata((Lat,Long), Elev, (lat_deg, long_deg), method='linear')        
    cartesian_pts      = np.dstack((np.dstack((x_pts[:,:,None],y_pts[:,:,None] )),z_deg[:,:,None])).reshape(ground_microphone_x_resolution*ground_microphone_y_resolution,3)
    lat_long_pts       = np.dstack((np.dstack((lat_deg[:,:,None],long_deg[:,:,None] )),z_deg[:,:,None])).reshape(ground_microphone_x_resolution*ground_microphone_y_resolution,3)
     
    # Compute distance between departure and destimation points
    coord0_rad = departure_coordinates*Units.degrees
    coord1_rad = destination_coordinates*Units.degrees  
    angle      = np.arccos(np.sin(coord0_rad[0])*np.sin(coord1_rad[0]) + 
                           np.cos(coord0_rad[0])*np.cos(coord1_rad[0])*np.cos(coord0_rad[1] - coord1_rad[1])) 
      
    # Compute heading from departure to destination    
    gamma = np.arcsin( np.sin(np.pi/2 - coord1_rad[0])* np.sin(coord1_rad[1] - coord0_rad[1])/np.sin(angle)) 
    angle_vector   = destination_coordinates - departure_coordinates 
    if angle_vector[0] < 0:
        gamma = np.pi - gamma 
    
    # Compute relative cartesian location of departure and destimation points on topographical grid 
    corner_long = lat_long_pts[0,1]
    if corner_long>180:
        corner_long = corner_long-360  
    
    lat_flag             = np.where(departure_coordinates<0)[0]
    departure_coordinates[lat_flag]  = departure_coordinates[lat_flag] + 360 
    long_flag            = np.where(destination_coordinates<0)[0]
    destination_coordinates[long_flag] = destination_coordinates[long_flag] + 360   

    
    # pack data 
    topography_data = Data(
        topography_file                         = topography_file,
        ground_microphone_x_resolution          = ground_microphone_x_resolution,
        ground_microphone_y_resolution          = ground_microphone_y_resolution,
        ground_microphone_x_stencil             = ground_microphone_x_stencil,
        ground_microphone_y_stencil             = ground_microphone_y_stencil,             
        ground_microphone_min_x                 = x_pts[0,0],               
        ground_microphone_max_x                 = x_pts[0,-1], 
        ground_microphone_min_y                 = y_pts[0,0],                   
        ground_microphone_max_y                 = y_pts[0,-1],    
        ground_microphone_min_lat               = x_min_coord,   
        ground_microphone_max_lat               = x_max_coord,  
        ground_microphone_min_long              = y_min_coord,  
        ground_microphone_max_long              = y_max_coord,  
        ground_microphone_locations             = cartesian_pts,
        ground_microphone_coordinates = lat_long_pts)
    
    return topography_data