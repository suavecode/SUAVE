## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# hess_smith.py 
# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np

from .panel_geometry import panel_geometry
from .infl_coeff  import infl_coeff
from .veldis import veldis

# ----------------------------------------------------------------------
# hess_smith.py
# ----------------------------------------------------------------------  

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def hess_smith(x,y,alpha,npanel):
    """Computes of the incompressible, inviscid flow over an airfoil of  arbitrary shape unp.sing the Hess-Smith panel method.  

    Assumptions:
    None

    Source:  "An introduction to theoretical and computational        
                    aerodynamics", J. Moran, Wiley, 1984  
 
                                                     
    Inputs                                                              
    alpha   -  Airfoil angle of attack                                  
    npanel  -  Number of panels on the airfoil.  The number of nodes  
                is equal to npanel+1, and the ith panel goes from node   
                i to node i+1                                
                                                                     
    Outputs                                                
    cl      -  Airfoil lift coefficient                   
    cd      -  Airfoil drag coefficient                
    cm      -  Airfoil moment coefficient about the c/4             
    x       -  Vector of x coordinates of the surface nodes        
    y       -  Vector of y coordinates of the surface nodes         
    cp      -  Vector of coefficients of pressure at the nodes     

    Properties Used:
    N/A
    """      
    # generate panel geometry data for later use 
    l    = np.zeros(npanel)
    st   = np.zeros(npanel)
    ct   = np.zeros(npanel)
    xbar = np.zeros(npanel)
    ybar = np.zeros(npanel)
    
    l,st,ct,xbar,ybar = panel_geometry(x,y,npanel)
     
    # compute matrix of aerodynamic influence coefficients  
    ainfl = np.zeros((npanel+1,npanel+1))
    ainfl = infl_coeff(x,y,xbar,ybar,st,ct,ainfl,npanel)
     
    # compute right hand side vector for the specified angle of attack 
    b  = np.zeros(npanel+1) 
    
    for i in range(npanel):
        b[i] = st[i]*np.cos(alpha)-np.sin(alpha)*ct[i]
     
    b[-1] = -(ct[0]*np.cos(alpha) + st[0]*np.sin(alpha))-(ct[-1]*np.cos(alpha) +st[-1]*np.sin(alpha))
               
    # solve matrix system for vector of q_i and gamma 
    b = np.atleast_2d(b)
    qg = np.linalg.inv(ainfl) * b
    
    # compute the tangential velocity distribution at the midpoint of panels 
    vt = veldis(qg,x,y,xbar,ybar,st,ct,alpha,npanel)
    
    return  xbar,ybar,vt,ct   