## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# 
# Created:  Aug 2015, SUAVE Team
# Modified: Jul 2020, M.Clarke
#           Sep 2020, M. Clarke 

from SUAVE.Core import Data  
import numpy as np

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def compute_naca_4series(airfoil_geometry_files,npoints=200):
    """Computes the points of NACA 4-series airfoil

    Assumptions:
    None

    Source:
    None

    Inputs:
        airfoils   - string of 4 digit NACA airfoils 

    Outputs:
    airfoil_data.
        thickness_to_chord 
        x_coordinates 
        y_coordinates
        x_upper_surface
        x_lower_surface
        y_upper_surface
        y_lower_surface
        camber_coordinates  

    Properties Used:
    N/A
    """         
 
    num_foils     = len(airfoil_geometry_files)   

    airfoil_data                    = Data() 
    airfoil_data.airfoil_names      = []        
    airfoil_data.x_coordinates      = []
    airfoil_data.y_coordinates      = []
    airfoil_data.thickness_to_chord = []
    airfoil_data.max_thickness      = []
    airfoil_data.camber_coordinates = []
    airfoil_data.x_upper_surface    = []
    airfoil_data.x_lower_surface    = []
    airfoil_data.y_upper_surface    = []
    airfoil_data.y_lower_surface    = []   
    
    for foil in range(num_foils):
        airfoil_data.airfoil_names.append(airfoil_geometry_files[foil])
        airfoil_digits  = [int(x) for x in airfoil_geometry_files[foil]] 
        camber          = airfoil_digits[0]/100 #   Maximum camber as a fraction of chord
        camber_loc      = airfoil_digits[1]/10  #   Maximum camber location as a fraction of chord
        thickness       = airfoil_digits[2]/10 + airfoil_digits[3]/100 #   Maximum thickness as a fraction of chord  
     
        N  = int(npoints/2)                         # number of points per side 
        te = 1.5                                    # points per side and trailing-edge bunching factor
        f  = np.linspace(0,1,N+1)                   # linearly-spaced points between 0 and 1
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
        
        airfoil_data.max_thickness.append(max_t)
        airfoil_data.x_coordinates.append(x_data)  
        airfoil_data.y_coordinates.append(y_data)     
        airfoil_data.x_upper_surface.append(x)
        airfoil_data.x_lower_surface.append(x)
        airfoil_data.y_upper_surface.append(np.append(0,y_up_surf))
        airfoil_data.y_lower_surface.append(y_lo_surf[::-1])          
        airfoil_data.camber_coordinates.append(camber)        
        airfoil_data.thickness_to_chord.append(t_c) 
    
    return airfoil_data 