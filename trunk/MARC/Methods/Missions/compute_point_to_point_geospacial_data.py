## @ingroup Methods-Missions
# compute_point_to_point_geospacial_data.py
# 
# Created: Mar 2023, M. Clarke  

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------
import MARC
from MARC.Core import Units, Data
from scipy.interpolate import griddata
import numpy as np
from geopy.distance import geodesic as GD

# ----------------------------------------------------------------------
#  Compute Point to Point Geospacial Data
# ---------------------------------------------------------------------
## @ingroup Methods-Missions
def compute_point_to_point_geospacial_data(topography_file                        = None,
                                            departure_tag                         = 'DEP',
                                            destination_tag                       = 'DES',
                                            departure_coordinates                 = [0.0,0.0],
                                            destination_coordinates               = [0.0,0.0]):
    """This computes the absolute microphone/observer locations on a defined topography
            
    Assumptions: 
        topography_file is a text file obtained from https://topex.ucsd.edu/cgi-bin/get_data.cgi
    
    Source:
        N/A  

    Inputs:   
        topography_file                        - file of lattide, longitude and elevation points     
        departure_coordinates                  - coordinates of departure location                                              [degrees]
        destination_coordinates                - coordinates of destimation location                                            [degrees]  
        
    Outputs:                                   
        latitude_longitude_micrphone_locations - latitude-longitude and elevation coordinates of all microphones in domain      [deg,deg,m]  
        flight_range                           - gound distance between departure and destination location                      [meters]              
        true_course_angle                      - true course angle measured clockwise from true north                     [radians]                      
        departure_location                     - cartesial coordinates of departure location relative to computational domain   [meters]                   
        destination_xyz_location                   - cartesial coordinates of destination location relative to computational domain [meters]    
    
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
    
    earth = MARC.Attributes.Planets.Earth()
    R     = earth.mean_radius      
     
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
 
    x_min_coord = np.min(Lat)
    y_min_coord = np.min(Long)
    dep_lat     = departure_coordinates[0]
    dep_long    = departure_coordinates[1]
    des_lat     = destination_coordinates[0]
    des_long    = destination_coordinates[1]
    if dep_long>180: 
        dep_long = dep_long-360
    if des_long>180:
        des_long = des_long-360  
    
    bottom_left_map_coords   = np.array([x_min_coord,y_min_coord])  
    x0_coord                 = np.array([dep_lat,y_min_coord])
    y0_coord                 = np.array([x_min_coord,dep_long])
    x1_coord                 = np.array([des_lat,y_min_coord])
    y1_coord                 = np.array([x_min_coord,des_long])  
    
    x0 = GD(x0_coord,bottom_left_map_coords).m 
    y0 = GD(y0_coord,bottom_left_map_coords).m  
    x1 = GD(x1_coord,bottom_left_map_coords).m 
    y1 = GD(y1_coord,bottom_left_map_coords).m 
    
    lat_flag             = np.where(departure_coordinates<0)[0]
    departure_coordinates[lat_flag]  = departure_coordinates[lat_flag] + 360 
    long_flag            = np.where(destination_coordinates<0)[0]
    destination_coordinates[long_flag] = destination_coordinates[long_flag] + 360   
    z0                   = griddata((Lat,Long), Elev, (np.array([departure_coordinates[0]]),np.array([departure_coordinates[1]])), method='linear')[0]
    z1                   = griddata((Lat,Long), Elev, (np.array([destination_coordinates[0]]),np.array([destination_coordinates[1]])), method='linear')[0] 
    dep_loc              = np.array([x0,y0,z0])
    des_loc              = np.array([x1,y1,z1])
    
    # pack data 
    geospacial_data = Data( 
        flight_range                            = distance,
        true_course_angle                       = gamma,
        departure_tag                           = departure_tag,
        departure_coordinates                   = departure_coordinates,
        departure_location                      = dep_loc,
        destination_tag                         = destination_tag,
        destination_coordinates                 = destination_coordinates ,
        destination_location                    = des_loc)
    
    return geospacial_data
