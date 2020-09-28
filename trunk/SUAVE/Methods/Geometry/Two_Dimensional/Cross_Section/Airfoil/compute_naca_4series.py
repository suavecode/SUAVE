## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
from math import sqrt, sin, cos, atan
import numpy as np

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def compute_naca_4series(camber,camber_loc,thickness,npoints=200):
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
    upper       [-] numpy array of x-y coordinates on the upper surface
    lower       [-] numpy array of x-y coordinates on the lower surface

    Properties Used:
    N/A
    """        

    half_pnts = int(npoints/2)
    
    upper = []
    lower = []
    
    for i in range(1,half_pnts):
        x = float(i) / float(half_pnts);
        x = x*sqrt(x)
        
        # lines
        zt,zc,th = compute_naca_4series_lines(x,camber,camber_loc,thickness)
        
        # upper surface
        xu = x  - zt*sin(th)
        zu = zc + zt*cos(th)
        upper.append([xu,zu])
        
        # lower surface
        xl = x  + zt*sin(th)
        zl = zc - zt*cos(th)
        lower.append([xl,zl])
    
    upper = [[0.0,0.0]] + upper + [[1.0,0.0]]
    lower = [[0.0,0.0]] + lower + [[1.0,0.0]]
    
    upper = np.array(upper)
    lower = np.array(lower)
    
    return upper, lower


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
    zt = thickness/0.2 * (  0.2969*sqrt(x) 
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
            
        th = atan( (zo - zc)/0.00001 )
        
    return zt,zc,th