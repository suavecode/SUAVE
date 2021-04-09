## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# 
# Created:  Aug 2015, SUAVE Team
# Modified: Jul 2020, M.Clarke
#           Sep 2020, M. Clarke 

from SUAVE.Core import Data  
import numpy as np

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def compute_naca_4series(camber,camber_loc,thickness,npoints=100):
    """Computes the points of NACA 4-series airfoil

    Assumptions:
    None

    Source:
    None

    Inputs:
    camber      [-] Maximum camber as a fraction of chord
    camber_loc  [-] Maximum camber location as a fraction of chord
    thickness   [-] Maximum thickness as a fraction of chord
    npoints     [-] Total number of points

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
    
    airfoil_data                    = Data()
    airfoil_data.x_coordinates      = []
    airfoil_data.y_coordinates      = []
    airfoil_data.thickness_to_chord = []
    airfoil_data.camber_coordinates = []
    airfoil_data.x_upper_surface    = []
    airfoil_data.x_lower_surface    = []
    airfoil_data.y_upper_surface    = []
    airfoil_data.y_lower_surface    = []
    
    half_pnts = int(npoints/2)
    
    upper = []
    lower = []
    
    for i in range(1,half_pnts):
        x = float(i) / float(half_pnts);
        x = x*np.sqrt(x)
        
        # lines
        zt,zc,th = compute_naca_4series_lines(x,camber,camber_loc,thickness)
        
        # upper surface
        xu = x  - zt*np.sin(th)
        zu = zc + zt*np.cos(th)
        upper.append([xu,zu])
        
        # lower surface
        xl = x  + zt*np.sin(th)
        zl = zc - zt*np.cos(th)
        lower.append([xl,zl])
    
    upper = [[0.0,0.0]] + upper + [[1.0,0.0]]
    lower = [[0.0,0.0]] + lower + [[1.0,0.0]]
    
    upper = np.array(upper)
    lower = np.array(lower) 
    
    x_up_surf = upper[:,0]
    x_lo_surf = lower[:,0]
    y_up_surf = upper[:,1]
    y_lo_surf = lower[:,1]  
     
    # compute thickness, camber and concatenate coodinates 
    thickness     = y_up_surf - y_lo_surf 
    camber        = y_lo_surf + thickness/2 
    x_data        = np.concatenate([x_up_surf[::-1],x_lo_surf])
    y_data        = np.concatenate([y_up_surf[::-1],y_lo_surf])  
    
    airfoil_data.thickness_to_chord.append(np.max(thickness))    
    airfoil_data.x_coordinates.append(x_data)  
    airfoil_data.y_coordinates.append(y_data)     
    airfoil_data.x_upper_surface.append(x_up_surf)
    airfoil_data.x_lower_surface.append(x_lo_surf)
    airfoil_data.y_upper_surface.append(y_up_surf)
    airfoil_data.y_lower_surface.append(y_lo_surf)          
    airfoil_data.camber_coordinates.append(camber)      
    
    return airfoil_data


## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def compute_naca_4series_lines(x,camber,camber_loc,thickness):
    """Computes the camber, thickness, and the angle of the camber line
    at a given point along the airfoil.

    Assumptions:
    None

    Source:
    Similar to http://airfoiltools.com/airfoil/naca4digit

    Inputs:
    camber      [-]       Maximum camber as a fraction of chord
    camber_loc  [-]       Maximum camber location as a fraction of chord
    thickness   [-]       Maximum thickness as a fraction of chord

    Outputs:
    zt          [-]       Thickness
    zc          [-]       Camber
    th          [radians] Angle of the camber line

    Properties Used:
    N/A
    """          

    xx = x*x

    # thickness
    zt = thickness/0.2 * (  0.2969*np.sqrt(x) 
                          - 0.1260*(x)
                          - 0.3516*(xx) 
                          + 0.2843*(x*xx) 
                          - 0.1015*(xx*xx)  )
    
    # symmetric
    if ( camber<=0.0 or camber_loc<=0.0 or camber_loc>=1.0 ):
        zc = 0.0
        th = 0.0

    else:
        
        # camber
        if x < camber_loc:
            zc = (camber/(camber_loc*camber_loc)) * \
                 (2.0*camber_loc*x - xx)
        else:
            zc = (camber/((1.0 - camber_loc)*(1.0 - camber_loc))) * \
                 (1.0 - 2.0*camber_loc + 2.0*camber_loc*x - xx)
        
        # finite difference theta
        xo = x + 0.00001;
        xoxo = xo*xo;        
        if xo < camber_loc:
            zo = (camber/(camber_loc*camber_loc)) * \
                 (2.0*camber_loc*xo - xoxo)
        else:
            zo = (camber/((1.0 - camber_loc)*(1.0 - camber_loc))) * \
                 (1.0 - 2.0*camber_loc + 2.0*camber_loc*xo - xoxo)
            
        th = np.arctan( (zo - zc)/0.00001 )
        
    return zt,zc,th