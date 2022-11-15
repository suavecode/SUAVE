## @ingroup Methods-Noise-Fidelity_One-Noise_Tools
# generate_microphone_points.py
# 
# Created: Sep 2021, M. Clarke  

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units, Data
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
def preprocess_ground_microphone_data(ground_microphone_min_x            = 0*Units.km,
                                      ground_microphone_max_x            = 10*Units.km,
                                      ground_microphone_min_y            = -5*Units.km,
                                      ground_microphone_max_y            = 5*Units.km,
                                      ground_microphone_x_resolution     = 101,
                                      ground_microphone_y_resolution     = 101,
                                      number_of_latitudinal_microphones  = 3,
                                      number_of_longitudinal_microphones = 3):
    """This computes the absolute microphone/observer locations on a defined topography
            
    Assumptions: 
        topography_file is a text file obtained from https://topex.ucsd.edu/cgi-bin/get_data.cgi
    
    Source:
        N/A  

    Inputs:  
        topography_file                        - file of lattide, longitude and elevation points                                [-]
        departure_coordinates                  - coordinates of departure location                                              [degrees]
        destination_coordinates                - coordinates of destimation location                                            [degrees]
        number_of_latitudinal_microphones      - number of points on computational domain in latitudal direction                [-]
        number_of_longitudinal_microphones     - number of points on computational domain in  longitidinal direction            [-] 
        number_of_latitudinal_microphones      - number of points in stencil in latitudal direction                             [-] 
        number_of_longitudinal_microphones     - number of points in stencil in in longitidinal direction                       [-] 
        
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
        flight_range                           - gound distance between departure and destination location                      [meters]              
        true_course                            - true course angle measured clockwise from true north                           [radians]                      
        departure_location                     - cartesial coordinates of departure location relative to computational domain   [meters]                   
        destination_location                   - cartesial coordinates of destination location relative to computational domain [meters]    
    
    Properties Used:
        N/A       
    """     
    N_x                   = ground_microphone_x_resolution 
    N_y                   = ground_microphone_y_resolution
    num_gm                = N_x*N_y
    gm_mic_locations      = np.zeros((num_gm,3))     
    min_x                 = ground_microphone_min_x         
    max_x                 = ground_microphone_max_x         
    min_y                 = ground_microphone_min_y         
    max_y                 = ground_microphone_max_y   
    x_coords_0            = np.repeat(np.linspace(min_x,max_x,N_x)[:,np.newaxis],N_y, axis = 1)
    y_coords_0            = np.repeat(np.linspace(min_y,max_y,N_y)[:,np.newaxis],N_x, axis = 1).T
    z_coords_0            = np.zeros_like(x_coords_0) 
    gm_mic_locations[:,0] = x_coords_0.reshape(num_gm)
    gm_mic_locations[:,1] = y_coords_0.reshape(num_gm)
    gm_mic_locations[:,2] = z_coords_0.reshape(num_gm)  
    
    # pack data 
    topography_data = Data(
        ground_microphone_x_resolution          = number_of_latitudinal_microphones,
        ground_microphone_y_resolution          = number_of_longitudinal_microphones,
        ground_microphone_x_stencil             = number_of_latitudinal_microphones,
        ground_microphone_y_stencil             = number_of_longitudinal_microphones,             
        ground_microphone_min_x                 = ground_microphone_min_x,             
        ground_microphone_max_x                 = ground_microphone_max_x,
        ground_microphone_min_y                 = ground_microphone_min_y,                 
        ground_microphone_max_y                 = ground_microphone_max_y,
        ground_microphone_locations             = gm_mic_locations)
    
    return topography_data

## @ingroup Methods-Noise-Fidelity_One-Noise_Tools  
def preprocess_topography_and_route_data(topography_file                       = None,
                                         departure_coordinates                 = [0.0,0.0],
                                         destination_coordinates               = [0.0,0.0],
                                         number_of_latitudinal_microphones     = 101,
                                         number_of_longitudinal_microphones    = 101,
                                         latitudinal_microphone_stencil_size   = 3,
                                         longitudinal_microphone_stencil_size  = 3 ):
    """This computes the absolute microphone/observer locations on a defined topography
            
    Assumptions: 
        topography_file is a text file obtained from https://topex.ucsd.edu/cgi-bin/get_data.cgi
    
    Source:
        N/A  

    Inputs:  
        topography_file                        - file of lattide, longitude and elevation points                                [-]
        departure_coordinates                  - coordinates of departure location                                              [degrees]
        destination_coordinates                - coordinates of destimation location                                            [degrees]
        number_of_latitudinal_microphones      - number of points on computational domain in latitudal direction                [-]
        number_of_longitudinal_microphones     - number of points on computational domain in  longitidinal direction            [-] 
        latitudinal_microphone_stencil_size    - number of points in stencil in latitudal direction                             [-] 
        longitudinal_microphone_stencil_size   - number of points in stencil in in longitidinal direction                       [-] 
        
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
        flight_range                           - gound distance between departure and destination location                      [meters]              
        true_course                            - true course angle measured clockwise from true north                           [radians]                      
        departure_location                     - cartesial coordinates of departure location relative to computational domain   [meters]                   
        destination_location                   - cartesial coordinates of destination location relative to computational domain [meters]    
    
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
    
    earth = SUAVE.Attributes.Planets.Earth()
    R     = earth.mean_radius      
    
    # interpolate data to defined x and y discretization
    x_dist_max         = (np.max(Lat)-np.min(Lat))*Units.degrees * R  
    y_dist_max         = (np.max(Long)-np.min(Long))*Units.degrees * R   
    [y_pts,x_pts]      = np.meshgrid(np.linspace(0,y_dist_max,number_of_longitudinal_microphones),np.linspace(0,x_dist_max,number_of_latitudinal_microphones))
    [long_deg,lat_deg] = np.meshgrid(np.linspace(np.min(Long),np.max(Long),number_of_longitudinal_microphones),np.linspace(np.min(Lat),np.max(Lat),number_of_latitudinal_microphones)) 
    z_deg              = griddata((Lat,Long), Elev, (lat_deg, long_deg), method='linear')        
    cartesian_pts      = np.dstack((np.dstack((x_pts[:,:,None],y_pts[:,:,None] )),z_deg[:,:,None])).reshape(number_of_latitudinal_microphones*number_of_longitudinal_microphones,3)
    lat_long_pts       = np.dstack((np.dstack((lat_deg[:,:,None],long_deg[:,:,None] )),z_deg[:,:,None])).reshape(number_of_latitudinal_microphones*number_of_longitudinal_microphones,3)
     
    # Compute distance between departure and destimation points
    coord0_rad = departure_coordinates*Units.degrees
    coord1_rad = destination_coordinates*Units.degrees  
    angle      = np.arccos(np.sin(coord0_rad[0])*np.sin(coord1_rad[0]) + 
                           np.cos(coord0_rad[0])*np.cos(coord1_rad[0])*np.cos(coord0_rad[1] - coord1_rad[1]))
    distance   = R*angle

    # Compute heading from departure to destination    
    gamma = np.arcsin( np.sin(np.pi/2 - coord1_rad[0])* np.sin(coord1_rad[1] - coord0_rad[1])/np.sin(angle)) 
    angle_vector   = destination_coordinates - departure_coordinates 
    if angle_vector[0] < 0:
        gamma = np.pi - gamma 
    
    # Compute relative cartesian location of departure and destimation points on topographical grid
    corner_lat  = lat_long_pts[0,0]
    corner_long = lat_long_pts[0,1]
    if corner_long>180:
        corner_long = corner_long-360  
    x0 = (coord0_rad[0] - corner_lat*Units.degrees)*R
    y0 = (coord0_rad[1] - corner_long*Units.degrees)*R  
    x1 = (coord1_rad[0] - corner_lat*Units.degrees)*R
    y1 = (coord1_rad[1] - corner_long*Units.degrees)*R 
    
    lat_flag             = np.where(departure_coordinates<0)[0]
    departure_coordinates[lat_flag]  = departure_coordinates[lat_flag] + 360 
    long_flag            = np.where(destination_coordinates<0)[0]
    destination_coordinates[long_flag] = destination_coordinates[long_flag] + 360   
    z0                   = griddata((Lat,Long), Elev, (np.array([departure_coordinates[0]]),np.array([departure_coordinates[1]])), method='linear')[0]
    z1                   = griddata((Lat,Long), Elev, (np.array([destination_coordinates[0]]),np.array([destination_coordinates[1]])), method='linear')[0] 
    dep_loc              = np.array([x0,y0,z0])
    des_loc              = np.array([x1,y1,z1])
    
    # pack data 
    topography_data = Data(
        ground_microphone_x_resolution          = number_of_latitudinal_microphones,
        ground_microphone_y_resolution          = number_of_longitudinal_microphones,
        ground_microphone_x_stencil             = latitudinal_microphone_stencil_size,
        ground_microphone_y_stencil             = longitudinal_microphone_stencil_size,             
        ground_microphone_min_x                 = x_pts[0,0],               
        ground_microphone_max_x                 = x_pts[0,-1], 
        ground_microphone_min_y                 = y_pts[0,0],                   
        ground_microphone_max_y                 = y_pts[0,-1],                      
        cartesian_microphone_locations          = cartesian_pts,
        latitude_longitude_microphone_locations = lat_long_pts, 
        flight_range                            = distance,
        true_course                             = gamma,
        departure_location                      = dep_loc,
        destination_location                    = des_loc)
    
    return topography_data
    
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
