## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# 
# Created:  Aug 2015, SUAVE Team
# Modified: Jul 2020, M.Clarke
#           Sep 2020, M. Clarke 

from SUAVE.Core import Data  
import math 
import numpy as np


## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def compute_naca_4series(airfoil_geometry_files,npoints= 200):
    """Computes the points of NACA 4-series airfoil

    Assumptions:
    None

    Source:
    None

    Inputs:
        airfoils              <string>

    Outputs:
    airfoil_data.
        thickness_to_chord    [unitless]
        x_coordinates         [meters]
        y_coordinates         [meters]
        x_upper_surface       [meters]
        x_lower_surface       [meters]
        y_upper_surface       [meters]
        y_lower_surface       [meters]
        camber_coordinates    [meters] 

    Properties Used:
    N/A
    """         
    
    num_airfoils                = len(airfoil_geometry_files)  
    half_npoints                = math.floor(npoints/2)      # number of points per side   
    
    geometry                    = Data()  
    geometry.airfoil_names      = airfoil_geometry_files         
    geometry.x_coordinates      = np.zeros((num_airfoils,npoints+ (npoints%2 == 0)))
    geometry.y_coordinates      = np.zeros((num_airfoils,npoints+ (npoints%2 == 0)))
    geometry.thickness_to_chord = np.zeros((num_airfoils,1))
    geometry.max_thickness      = np.zeros((num_airfoils,1))
    geometry.camber_coordinates = np.zeros((num_airfoils,half_npoints+1))
    geometry.x_upper_surface    = np.zeros((num_airfoils,half_npoints+1))
    geometry.x_lower_surface    = np.zeros((num_airfoils,half_npoints+1))
    geometry.y_upper_surface    = np.zeros((num_airfoils,half_npoints+1))
    geometry.y_lower_surface    = np.zeros((num_airfoils,half_npoints+1)) 
     
    for i in range(num_airfoils): 
        airfoil_digits  = [int(x) for x in airfoil_geometry_files[i]] 
        camber          = airfoil_digits[0]/100 #   Maximum camber as a fraction of chord
        camber_loc      = airfoil_digits[1]/10  #   Maximum camber location as a fraction of chord
        thickness       = airfoil_digits[2]/10 + airfoil_digits[3]/100 #   Maximum thickness as a fraction of chord  
     
        te = 1.5                                    # points per side and trailing-edge bunching factor
        f  = np.linspace(0,1,half_npoints+1)        # linearly-spaced points between 0 and 1
        x  = 1 - (te+1)*f*(1-f)**te - (1-f)**(te+1) # bunched points, x, 0 to 1
    
        # normalized thickness, gap at trailing edge (use -.1036*x^4 for no gap)
        t    = .2969*np.sqrt(x) - 0.126*x - 0.3516*(x**2) + 0.2843*(x**3) - 0.1015*(x**4)
        t    = t*thickness/.2
        m    = camber
        p    = camber_loc
        c    = m/(1-p)**2 * ((1-2*p)+2*p*x-x**2)
        
        if m == 0 and p == 0:
            pass
        else:
            I    = np.where(x<p)[0]  
            c[I] = m/p**2*(2*p*x[I]-x[I]**2) 
        
        x_up_surf = x[1:]
        y_up_surf = (c + t)[1:]
        x_lo_surf = np.flip(x)
        y_lo_surf = np.flip(c - t)
       
        # concatenate upper and lower surfaces  
        x_data    = np.hstack((x_lo_surf ,x_up_surf))
        y_data    = np.hstack((y_lo_surf, y_up_surf))  

        max_t  = np.max(thickness)
        max_c  = max(x_data) - min(x_data)
        t_c    = max_t/max_c         
        
        geometry.max_thickness[i]      = max_t 
        geometry.x_coordinates[i]      = x_data   
        geometry.y_coordinates[i]      = y_data      
        geometry.x_upper_surface[i]    = x 
        geometry.x_lower_surface[i]    = x 
        geometry.y_upper_surface[i]    = np.append(0,y_up_surf) 
        geometry.y_lower_surface[i]    = y_lo_surf[::-1]           
        geometry.camber_coordinates[i] = camber         
        geometry.thickness_to_chord[i] = t_c 
    
    return geometry