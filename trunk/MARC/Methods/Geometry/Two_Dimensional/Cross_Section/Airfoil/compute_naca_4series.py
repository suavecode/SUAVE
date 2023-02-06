## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# compute_naca_4series.py
#
# Created:  Aug 2015, SUAVE Team (Stanford University)
# Modified: Jul 2020, M.Clarke
#           Sep 2020, M. Clarke 

from MARC.Core import Data   
import numpy as np


## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def compute_naca_4series(airfoil_geometry_file,npoints= 200, leading_and_trailing_edge_resolution_factor = 1.5 ):
    """Computes the points of NACA 4-series airfoil

    Assumptions:
    None

    Source:
    None

    Inputs:
        airfoils                                     <string>
        npoints                                      [unitless]    
        leading_and_trailing_edge_resolution_factor  [unitless]

    Outputs:
    airfoil_data.
        thickness_to_chord                           [unitless]
        x_coordinates                                [meters]
        y_coordinates                                [meters]
        x_upper_surface                              [meters]
        x_lower_surface                              [meters]
        y_upper_surface                              [meters]
        y_lower_surface                              [meters]
        camber_coordinates                           [meters] 

    Properties Used:
    N/A
    """         
    geometry        = Data()
    half_npoints    = npoints/2                                    # number of points per side    
    airfoil_digits  = [int(x) for x in airfoil_geometry_file]
    camber          = airfoil_digits[0]/100                        # maximum camber as a fraction of chord
    camber_loc      = airfoil_digits[1]/10                         # maximum camber location as a fraction of chord
    thickness       = airfoil_digits[2]/10 + airfoil_digits[3]/100 # maximum thickness as a fraction of chord  
    
    x_us  = np.linspace(0,1,int(np.ceil(half_npoints))+ (npoints%2 == 0))  
    x_ls  = np.linspace(0,1,int(np.ceil(half_npoints)))  
    if leading_and_trailing_edge_resolution_factor != None: 
        te = leading_and_trailing_edge_resolution_factor    # points per side and trailing-edge bunching factor 
        x_us  = 1 - (te+1)*x_us*(1-x_us)**te - (1-x_us)**(te+1)         # bunched points, x, 0 to 1
        x_ls  = 1 - (te+1)*x_ls*(1-x_ls)**te - (1-x_ls)**(te+1)         # bunched points, x, 0 to 1

    # normalized thickness, gap at trailing edge  
    t_us = .2969*np.sqrt(x_us) - 0.126*x_us - 0.3516*(x_us**2) + 0.2843*(x_us**3) - 0.1015*(x_us**4)
    t_ls = .2969*np.sqrt(x_ls) - 0.126*x_ls - 0.3516*(x_ls**2) + 0.2843*(x_ls**3) - 0.1015*(x_ls**4)
    t_us = t_us*thickness/.2
    t_ls = t_ls*thickness/.2
    m    = camber
    p    = camber_loc
    c_us = m/(1-p)**2 * ((1-2*p)+2*p*x_us-x_us**2)
    c_ls = m/(1-p)**2 * ((1-2*p)+2*p*x_ls-x_ls**2)
    
    if m == 0 and p == 0:
        pass
    else:
        I_us = np.where(x_us<p)[0] 
        I_ls = np.where(x_ls<p)[0]  
        c_us[I_us] = m/p**2*(2*p*x_us[I_us]-x_us[I_us]**2) 
        c_ls[I_ls] = m/p**2*(2*p*x_ls[I_ls]-x_ls[I_ls]**2) 
    
    x_up_surf = x_us[1:]
    x_lo_surf = np.flip(x_ls)
    y_up_surf = (c_us + t_us)[1:] 
    y_lo_surf = np.flip(c_ls - t_ls)
   
    # concatenate upper and lower surfaces  
    x_data = np.hstack((x_lo_surf,x_up_surf))
    y_data = np.hstack((y_lo_surf, y_up_surf))  

    max_t  = np.max(thickness)
    max_c  = max(x_data) - min(x_data)
    t_c    = max_t/max_c         
    
    geometry.max_thickness      = max_t 
    geometry.x_coordinates      = x_data   
    geometry.y_coordinates      = y_data      
    geometry.x_upper_surface    = x_us 
    geometry.x_lower_surface    = x_ls 
    geometry.y_upper_surface    = np.append(0,y_up_surf) 
    geometry.y_lower_surface    = y_lo_surf[::-1]           
    geometry.camber_coordinates = c_ls      
    geometry.thickness_to_chord = t_c 
    
    return geometry