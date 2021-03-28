## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# veldis.py 
#
# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
# veldis
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def veldis(qg,x,y,xbar,ybar,st,ct,al,npanel):
    """Compute the tangential velocity distribution at the       
                 midpoint of each panel   

    Assumptions:
    None  
    
    Inputs:                                                    

     qg          -  Vector of source/np.sink and vortex strengths              
     x            -  Vector of x coordinates of the surface nodes            
     y            -  Vector of y coordinates of the surface nodes             
     xbar      -  X-coordinate of the midpoint of each panel            
     ybar      -  X-coordinate of the midpoint of each panel            
     st          -  np.sin(theta) for each panel                                
     ct          -  np.cos(theta) for each panel                               
     al          -  Angle of attack in radians                              
     npanel  -  Number of panels on the airfoil                 

     Outputs:                                                        

     vt      -  Vector of tangential velocities      

    Properties Used:
    N/A
    """     
 
    # flow tangency boundary condition - source distribution 
    vt    = np.zeros(npanel)
    gamma = qg[-1]
    
    # convert 1d matrices to 2d  
    qg_2d   = np.repeat(np.atleast_2d(qg[:-1]).T,npanel, axis = 0) 
    x_2d    = np.repeat(np.atleast_2d(x),npanel, axis = 0) 
    y_2d    = np.repeat(np.atleast_2d(y),npanel, axis = 0)
    xbar_2d = np.repeat(np.atleast_2d(xbar).T,npanel, axis = 1)
    ybar_2d = np.repeat(np.atleast_2d(ybar).T,npanel, axis = 1)
    st_2d   = np.atleast_2d(st)
    ct_2d   = np.atleast_2d(ct)  
    
    vt                   = ct *np.cos(al) + st *np.sin(al)
    sti_minus_j          = np.dot(st_2d.T,ct_2d) - np.dot(ct_2d.T,st_2d)
    cti_minus_j          = np.dot(ct_2d.T,ct_2d) + np.dot(st_2d.T,st_2d)
    rij                  = np.sqrt((xbar_2d-x_2d[:,:-1])**2 + (ybar_2d-y_2d[:,:-1])**2)
    rij_plus_1           = np.sqrt((xbar_2d-x_2d[:,1:])**2 +  (ybar_2d-y_2d[:,1:])**2)
    rij_dot_rij_plus_1   = (xbar_2d-x_2d[:,:-1])*(xbar_2d-x_2d[:,1:]) + (ybar_2d-y_2d[:,:-1])*(ybar_2d-y_2d[:,1:])  
    anglesign            = np.sign((xbar_2d-x_2d[:,:-1])*(ybar_2d-y_2d[:,1:]) - (xbar_2d-x_2d[:,1:])*(ybar_2d-y_2d[:,:-1]))
    betaij               = np.real(anglesign*np.arccos(rij_dot_rij_plus_1/rij/rij_plus_1))
    diag_indices         = np.diag_indices(npanel)
    betaij[diag_indices] = np.pi     
    
    vt += np.sum(qg_2d/2/np.pi*(sti_minus_j*betaij - cti_minus_j*np.log(rij_plus_1/rij)),1)  + \
          np.sum(gamma/2/np.pi*(sti_minus_j*np.log(rij_plus_1/rij) + cti_minus_j*betaij),1)
    
    return  vt  
