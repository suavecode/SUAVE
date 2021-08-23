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
def panel_geometry(x,y,npanel,nalpha,nRe):
    """Computes airfoil surface panelization parameters for later use in 
    the computation of the matrix of influence coefficients.        

    Assumptions:
    None

    Source:
    None 
                                                                       
    Inputs:                                                         
    x       -  Vector of x coordinates of the surface nodes  [unitless]         
    y       -  Vector of y coordinates of the surface nodes  [unitless]      
    npanel  -  Number of panels on the airfoil               [unitless]         
                                                                     
    Outputs:                                             
    l       -  Panel lengths                              [unitless]
    st      -  np.sin(theta) for each panel               [radians]
    ct      -  np.cos(theta) for each panel               [radians]
    xbar    -  x-coordinate of the midpoint of each panel [unitless]              
    ybar    -  y-coordinate of the midpoint of each panel [unitless]               
    
    
    Properties Used:
    N/A
    """     
    # compute various geometrical quantities    
    l    = np.sqrt((x[1:] -x[:-1])**2 +(y[1:] -y[:-1])**2)
    st   = (y[1:] -y[:-1])/l 
    ct   = (x[1:] -x[:-1])/l 
    xbar = (x[1:] +x[:-1])/2
    ybar = (y[1:] +y[:-1])/2 
    
    norm  = np.zeros((npanel,2,nalpha,nRe))
    norm[:,0,:,:]  =  -st
    norm[:,1,:,:]  =  ct 
    
    return l,st,ct,xbar,ybar,norm 
     