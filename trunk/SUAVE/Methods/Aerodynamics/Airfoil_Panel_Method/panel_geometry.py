## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# panel_geometry.py 

# Created:  Mar 2021, M. Clarke

# ---------------------------------------
#-------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
# panel_geometry.py
# ----------------------------------------------------------------------  
## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def panel_geometry(x,y,npanel):
    """Compute airfoil surface panelization parameters  
                        for later use in the computation of the matrix 
                        of influence coefficients.        

    Assumptions:
    None

    Source:
    None 
                                                                       
    Inputs:                                                         
    x       -  Vector of x coordinates of the surface nodes      
    y       -  Vector of y coordinates of the surface nodes       
    npanel  -  Number of panels on the airfoil                       
                                                                     
    Outputs:                                             
    l       -  Panel lenghts                         
    st      -  np.sin(theta) for each panel  
    ct      -  np.cos(theta) for each panel  
    xbar    -  X-coordinate of the midpoint of each panel               
    ybar    -  X-coordinate of the midpoint of each panel                
    
    
    Properties Used:
    N/A
    """     
    # compute various geometrical quantities 
    l    = np.zeros(npanel)
    st   = np.zeros_like(l)
    ct   = np.zeros_like(l)
    xbar = np.zeros_like(l)
    ybar = np.zeros_like(l) 
    
    for i in range(npanel):
        l[i]    = np.sqrt((x[i+1] -x[i])**2 +(y[i+1] -y[i])**2)
        st[i]   = (y[i+1] -y[i])/l[i]
        ct[i]   = (x[i+1] -x[i])/l[i]
        xbar[i] = (x[i+1] +x[i])/2
        ybar[i] = (y[i+1] +y[i])/2 
    
    return l,st,ct,xbar,ybar
     