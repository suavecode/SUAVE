## @ingroup Methods-Missions
# compute_ground_distance_and_true_course_angle.py
# 
# Created: Nov 2022, M. Clarke

# ----------------------------------------------------------------------
#  Inports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Core import Units
## @ingroup Methods-Missions 
def compute_ground_distance_and_true_course_angle(departure_coordinates,destination_coordinates):
    """This computes gound distance and the true course angle between two points
            
    Assumptions:
        1) The earth is a perfect sphere with radius of earth = 6378.1 km  

    Source:
        N/A  

    Inputs:   
        departure_coordinates    - lattide and longitude cooridates of departure location   [degrees]
        destination_coordinates  - lattide and longitude cooridates of destination location [degrees]
        
    Outputs: 
        cartesian_pts  - cartesian coordiates (x,y,z) of all microphones in domain                [meters]
        lat_long_pts   - latitude-longitude and elevation coordiates of all microphones in domain [deg,deg,m] 
    
    Properties Used:
        N/A       
    """      
    earth            = SUAVE.Attributes.Planets.Earth()
    R                = earth.mean_radius     
    
    # Compute coordinates of departure and destination points to determine range 
    coord0_rad     = departure_coordinates*Units.degrees
    coord1_rad     = destination_coordinates*Units.degrees  
    angle          = np.arccos(np.sin(coord0_rad[0])*np.sin(coord1_rad[0]) + 
                     np.cos(coord0_rad[0])*np.cos(coord1_rad[0])*np.cos(coord0_rad[1] - coord1_rad[1]))
    aircraft_range = R*angle
        
    # Compute heading from departure to destination    
    true_course = np.arcsin( np.sin(np.pi/2 - coord1_rad[0])* np.sin(coord1_rad[1] - coord0_rad[1])/np.sin(angle))
    
    angle_vector = destination_coordinates - departure_coordinates 
    if angle_vector[0] < 0:
        true_course = np.pi - true_course
        
    return aircraft_range,true_course
    