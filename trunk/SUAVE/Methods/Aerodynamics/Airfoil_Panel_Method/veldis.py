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
def veldis(qg,x,y,xbar,ybar,st,ct,alpha,Re,npanel):
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
    
    nalpha           = len(alpha)
    nRe              = len(Re) 
    
    # flow tangency boundary condition - source distribution  
    vt_2d = ct *np.cos(alpha) + st*np.sin(alpha)
    gamma = np.repeat(qg[-1,:,:][np.newaxis,:,:],npanel, axis = 0)  
    
    # convert 1d matrices to 2d 
    qg_2d                = np.repeat(qg[:-1,:,:][np.newaxis,:,:,:],npanel, axis = 0) 
    x_2d                 = np.repeat(np.swapaxes(np.swapaxes(x,0, 2),0,1)[:,:,np.newaxis,:],npanel, axis = 2)
    y_2d                 = np.repeat(np.swapaxes(np.swapaxes(y,0, 2),0,1)[:,:,np.newaxis,:],npanel, axis = 2)
    xbar_2d              = np.repeat(np.swapaxes(np.swapaxes(xbar,0, 2),0,1)[:,:,:,np.newaxis],npanel, axis = 3)
    ybar_2d              = np.repeat(np.swapaxes(np.swapaxes(ybar,0, 2),0,1)[:,:,:,np.newaxis],npanel, axis = 3) 
    st_2d                = np.repeat(np.swapaxes(np.swapaxes(st,0, 2),0,1)[:,:,:,np.newaxis],npanel, axis = 3) 
    ct_2d                = np.repeat(np.swapaxes(np.swapaxes(ct,0, 2),0,1)[:,:,:,np.newaxis],npanel, axis = 3) 
    st_2d_T              = np.swapaxes(st_2d,2,3)
    ct_2d_T              = np.swapaxes(ct_2d,2,3)  
    
    # Fill the elements of the matrix of aero influence coefficients
    sti_minus_j          = ct_2d_T*st_2d - st_2d_T*ct_2d 
    cti_minus_j          = ct_2d_T*ct_2d + st_2d_T*st_2d 
    rij                  = np.sqrt((xbar_2d-x_2d[:,:,:,:-1])**2 + (ybar_2d-y_2d[:,:,:,:-1])**2)
    rij_plus_1           = np.sqrt((xbar_2d-x_2d[:,:,:,1:])**2 +  (ybar_2d-y_2d[:,:,:,1:])**2)
    rij_dot_rij_plus_1   = (xbar_2d-x_2d[:,:,:,:-1])*(xbar_2d-x_2d[:,:,:,1:]) + (ybar_2d-y_2d[:,:,:,:-1])*(ybar_2d-y_2d[:,:,:,1:])  
    anglesign            = np.sign((xbar_2d-x_2d[:,:,:,:-1])*(ybar_2d-y_2d[:,:,:,1:]) - (xbar_2d-x_2d[:,:,:,1:])*(ybar_2d-y_2d[:,:,:,:-1]))
    r_ratio              = rij_dot_rij_plus_1/rij/rij_plus_1
    r_ratio[r_ratio>1.0] = 1.0 # numerical noise     
    betaij               = np.real(anglesign*np.arccos(r_ratio))    
    diag_indices         =  list(np.tile(np.repeat(np.arange(npanel),nalpha),nRe))
    aoas                 = list(np.tile(np.arange(nalpha),nRe*npanel))
    res                  = list(np.repeat(np.arange(nRe),nalpha*npanel))   
    betaij[aoas,res,diag_indices,diag_indices] = np.pi      
    
    # swap axes 
    sti_minus_j_2d  = np.swapaxes(np.swapaxes(sti_minus_j,0,2),1,3)
    betaij_2d       = np.swapaxes(np.swapaxes(betaij,0,2),1,3)
    cti_minus_j_2d  = np.swapaxes(np.swapaxes(cti_minus_j,0,2),1,3)
    rij_2d          = np.swapaxes(np.swapaxes(rij,0,2),1,3)
    rij_plus_1_2d   = np.swapaxes(np.swapaxes(rij_plus_1,0,2),1,3)
     
    vt_2d += np.sum(qg_2d/2/np.pi*(sti_minus_j_2d*betaij_2d - cti_minus_j_2d*np.log(rij_plus_1_2d/rij_2d)),1)  + \
             np.sum(gamma/2/np.pi*(sti_minus_j_2d*np.log(rij_plus_1_2d/rij_2d) + cti_minus_j_2d*betaij_2d),1)
    
    return  vt_2d