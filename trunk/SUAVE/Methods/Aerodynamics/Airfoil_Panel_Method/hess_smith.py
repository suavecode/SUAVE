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
from .velocity_distribution import velocity_distribution

# ----------------------------------------------------------------------
# hess_smith.py
# ----------------------------------------------------------------------  

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def hess_smith(x_coord,y_coord,alpha,Re,npanel,batch_analyis):
    """Computes the incompressible, inviscid flow over an airfoil of  arbitrary shape using the Hess-Smith panel method.  

    Assumptions:
    None

    Source:  "An introduction to theoretical and computational        
                    aerodynamics", J. Moran, Wiley, 1984  
 
                                                     
    Inputs          
    x             -  Vector of x coordinates of the surface                  [unitess]     
    y             -  Vector of y coordinates of the surface                  [unitess] 
    batch_analyis - flag for batch analysis                                  [boolean]
    alpha         -  Airfoil angle of attack                                 [radians] 
    npanel        -  Number of panels on the airfoil.  The number of nodes   [unitess] 
                      is equal to npanel+1, and the ith panel goes from node   
                      i to node i+1                                
                                                                           
    Outputs                                                      
    cl            -  Airfoil lift coefficient                         [unitless]           
    cd            -  Airfoil drag coefficient                         [unitless]      
    cm            -  Airfoil moment coefficient about the c/4         [unitless]            
    x_bar         -  Vector of x coordinates of the surface nodes     [unitless]           
    y_bar         -  Vector of y coordinates of the surface nodes     [unitless]          
    cp            -  Vector of coefficients of pressure at the nodes  [unitless]         

    Properties Used:
    N/A
    """      
    
    nalpha        = len(alpha)
    nRe           = len(Re) 
    alpha_2d      = np.repeat(np.repeat(alpha,nRe, axis = 1)[np.newaxis,:, :], npanel, axis=0) 
    
    # generate panel geometry data for later use   
    l,st,ct,xbar,ybar,norm = panel_geometry(x_coord,y_coord,npanel,nalpha,nRe) 
    
    # compute matrix of aerodynamic influence coefficients
    ainfl         = infl_coeff(x_coord,y_coord,xbar,ybar,st,ct,npanel,nalpha,nRe,batch_analyis) # nalpha x nRe x npanel+1 x npanel+1
    
    # compute right hand side vector for the specified angle of attack 
    b_2d          = np.zeros((npanel+1,nalpha, nRe))
    b_2d[:-1,:,:] = st*np.cos(alpha_2d) - np.sin(alpha_2d)*ct
    b_2d[-1,:,:]  = -(ct[0,:,:]*np.cos(alpha_2d[-1,:,:]) + st[0,:,:]*np.sin(alpha_2d[-1,:,:]))-(ct[-1,:,:]*np.cos(alpha_2d[-1,:,:]) +st[-1,:,:]*np.sin(alpha_2d[-1,:,:]))
      
    # solve matrix system for vector of q_i and gamma  
    qg_T          = np.linalg.solve(ainfl,np.swapaxes(b_2d.T,0,1))
    qg            = np.swapaxes(qg_T.T,1,2) 
    
    # compute the tangential velocity distribution at the midpoint of panels 
    vt            = velocity_distribution(qg,x_coord,y_coord,xbar,ybar,st,ct,alpha,Re,npanel)
    
    return  xbar,ybar,vt,norm 