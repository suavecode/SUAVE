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
def veldis_old(qg,x,y,xbar,ybar,st,ct,alpha,Re,npanel):
    """Compute the tangential velocity distribution at the       
                 midpoint of each panel   
    Assumptions:
    None  
    
    Inputs:                                                    
     qg          -  Vector of source/sink and vortex strengths              
     x           -  Vector of x coordinates of the surface nodes            
     y           -  Vector of y coordinates of the surface nodes             
     xbar        -  x-coordinate of the midpoint of each panel            
     ybar        -  y-coordinate of the midpoint of each panel            
     st          -  np.sin(theta) for each panel                                
     ct          -  np.cos(theta) for each panel                               
     al          -  Angle of attack in radians                              
     npanel      -  Number of panels on the airfoil                 
     Outputs:                                                        
     vt          -  Vector of tangential velocities      
    Properties Used:
    N/A
    """     
    nalpha   = len(alpha)
    nRe      = len(Re)
    
    # flow tangency boundary condition - source distribution  
    gamma = np.repeat(qg[-1,:,:][np.newaxis,:,:],npanel, axis = 0)
    
    # convert 1d matrices to 2d  
    qg_2d   = np.repeat(qg[:-1,:,:][np.newaxis,:,:,:],npanel, axis = 0) 
    x_2d    = np.repeat(np.atleast_2d(x)         ,npanel, axis = 0) 
    y_2d    = np.repeat(np.atleast_2d(y)         ,npanel, axis = 0)
    xbar_2d = np.repeat(np.atleast_2d(xbar).T    ,npanel, axis = 1)
    ybar_2d = np.repeat(np.atleast_2d(ybar).T    ,npanel, axis = 1)
    st_2d   = np.atleast_2d(st)
    ct_2d   = np.atleast_2d(ct) 
    
    vt                   = ct *np.cos(alpha) + st*np.sin(alpha)
    sti_minus_j          = np.dot(st_2d.T,ct_2d) - np.dot(ct_2d.T,st_2d)
    cti_minus_j          = np.dot(ct_2d.T,ct_2d) + np.dot(st_2d.T,st_2d)
    rij                  = np.sqrt((xbar_2d-x_2d[:,:-1])**2 + (ybar_2d-y_2d[:,:-1])**2)
    rij_plus_1           = np.sqrt((xbar_2d-x_2d[:,1:])**2 +  (ybar_2d-y_2d[:,1:])**2)
    rij_dot_rij_plus_1   = (xbar_2d-x_2d[:,:-1])*(xbar_2d-x_2d[:,1:]) + (ybar_2d-y_2d[:,:-1])*(ybar_2d-y_2d[:,1:])  
    anglesign            = np.sign((xbar_2d-x_2d[:,:-1])*(ybar_2d-y_2d[:,1:]) - (xbar_2d-x_2d[:,1:])*(ybar_2d-y_2d[:,:-1]))
    r_ratio              = rij_dot_rij_plus_1/rij/rij_plus_1
    r_ratio[r_ratio>1.0] = 1.0 # numerical noise     
    betaij               = np.real(anglesign*np.arccos(r_ratio))
    diag_indices         = np.diag_indices(npanel)
    betaij[diag_indices] = np.pi     
    
    # convert to 2d 
    sti_minus_j_2d  = np.repeat(np.repeat(sti_minus_j[:, :,np.newaxis],nalpha, axis = 2)[:,:,:,np.newaxis] , nRe , axis = 3)
    betaij_2d       = np.repeat(np.repeat(betaij[:, :,np.newaxis]     ,nalpha, axis = 2)[:,:,:,np.newaxis] , nRe , axis = 3)
    cti_minus_j_2d  = np.repeat(np.repeat(cti_minus_j[:, :,np.newaxis],nalpha, axis = 2)[:,:,:,np.newaxis] , nRe , axis = 3)
    rij_2d          = np.repeat(np.repeat(rij[:, :,np.newaxis]        ,nalpha, axis = 2)[:,:,:,np.newaxis] , nRe , axis = 3)
    rij_plus_1_2d   = np.repeat(np.repeat(rij_plus_1[:, :,np.newaxis] ,nalpha, axis = 2)[:,:,:,np.newaxis] , nRe , axis = 3)
    
    vt_2d  = np.repeat(vt.T[:, :,np.newaxis],nRe, axis = 2)
    vt_2d += np.sum(qg_2d/2/np.pi*(sti_minus_j_2d*betaij_2d - cti_minus_j_2d*np.log(rij_plus_1_2d/rij_2d)),1)  + \
             np.sum(gamma/2/np.pi*(sti_minus_j_2d*np.log(rij_plus_1_2d/rij_2d) + cti_minus_j_2d*betaij_2d),1)
    
    return  vt_2d  