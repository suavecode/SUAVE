## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# hess_smith.py 
# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np

from .panel_geometry_old import panel_geometry_old
from .infl_coeff_old  import infl_coeff_old
from .veldis_old import veldis_old

# ----------------------------------------------------------------------
# hess_smith.py
# ----------------------------------------------------------------------  

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def hess_smith_old(x_coord,y_coord,alpha,Re,npanel):
    """Computes of the incompressible, inviscid flow over an airfoil of  arbitrary shape using the Hess-Smith panel method.  
    Assumptions:
    None
    Source:  "An introduction to theoretical and computational        
                    aerodynamics", J. Moran, Wiley, 1984  
 
                                                     
    Inputs          
    x       -  Vector of x coordinates of the surface         
    y       -  Vector of y coordinates of the surface      
    alpha   -  Airfoil angle of attack                                  
    npanel  -  Number of panels on the airfoil.  The number of nodes  
                is equal to npanel+1, and the ith panel goes from node   
                i to node i+1                                
                                                                     
    Outputs                                                
    cl      -  Airfoil lift coefficient                   
    cd      -  Airfoil drag coefficient                
    cm      -  Airfoil moment coefficient about the c/4             
    x_bar   -  Vector of x coordinates of the surface nodes        
    y_bar   -  Vector of y coordinates of the surface nodes         
    cp      -  Vector of coefficients of pressure at the nodes     
    Properties Used:
    N/A
    """      
    
    nalpha        = len(alpha)
    nRe           = len(Re) 
    alpha_2d      = np.repeat(np.repeat(alpha,nRe, axis = 1)[np.newaxis,:, :], npanel, axis=0) 
    
    # generate panel geometry data for later use   
    l,st,ct,xbar,ybar,norm = panel_geometry_old(x_coord,y_coord,npanel)
    
    # convert 1D vectors to 2D
    ct_2d         = np.repeat(np.repeat(ct[:,np.newaxis], nalpha, axis=1)[:,:,np.newaxis], nRe, axis=2)
    st_2d         = np.repeat(np.repeat(st[:,np.newaxis], nalpha, axis=1)[:,:,np.newaxis], nRe, axis=2) 
    
    # compute matrix of aerodynamic influence coefficients
    ainfl         = infl_coeff_old(x_coord,y_coord,xbar,ybar,st,ct,npanel)
    ainfl_2d      = np.repeat(np.repeat(ainfl[:, :,np.newaxis], nalpha, axis=2)[:, :, :, np.newaxis], nRe, axis=3)
    
    # compute right hand side vector for the specified angle of attack 
    b_2d          = np.zeros((npanel+1,nalpha, nRe))
    b_2d[:-1,:,:] = st_2d*np.cos(alpha_2d) - np.sin(alpha_2d)*ct_2d
    b_2d[-1,:,:]  = -(ct_2d[0,:,:]*np.cos(alpha_2d[-1,:,:]) + st_2d[0,:,:]*np.sin(alpha_2d[-1,:,:]))-(ct_2d[-1,:,:]*np.cos(alpha_2d[-1,:,:]) +st_2d[-1,:,:]*np.sin(alpha_2d[-1,:,:]))
      
    # solve matrix system for vector of q_i and gamma  
    qg_T          = np.linalg.solve(np.swapaxes(ainfl_2d.T,2,3),b_2d.T)
    qg            = qg_T.T
    
    # compute the tangential velocity distribution at the midpoint of panels 
    vt            = veldis_old(qg,x_coord,y_coord,xbar,ybar,st,ct,alpha,Re,npanel)
    
    return  xbar,ybar,vt,ct,norm 