## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# infl_coeff.py  

# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
# infl_coeff.py
# ----------------------------------------------------------------------  
## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def infl_coeff(x,y,xbar,ybar,st,ct,npanel,nalpha,nRe,batch_analyis):
    """Compute the matrix of aerodynamic influence  coefficients for later use

    Assumptions:
    None

    Source:
    None 
 
    Inputs
    x       -  Vector of x coordinates of the surface nodes  [unitless]   
    y       -  Vector of y coordinates of the surface nodes  [unitless]   
    xbar    -  x-coordinate of the midpoint of each panel    [unitless]      
    ybar    -  y-coordinate of the midpoint of each panel    [unitless]     
    st      -  np.sin(theta) for each panel                  [radians]               
    ct      -  np.cos(theta) for each panel                  [radians]                 
    npanel  -  Number of panels on the airfoil               [unitless]       
                                                                                            
    Outputs                                        
    ainfl   -  Aero influence coefficient matrix             [unitless]

    Properties Used:
    N/A
    """                          
   
    ainfl                = np.zeros((nalpha,nRe,npanel+1,npanel+1))    
    pi2inv               = 1 / (2*np.pi) 
    
    # convert 1d matrices to 4d 
    x_2d                 = np.repeat(np.swapaxes(np.swapaxes(x,0, 2),0,1)[:,:,np.newaxis,:],npanel, axis = 2)
    y_2d                 = np.repeat(np.swapaxes(np.swapaxes(y,0, 2),0,1)[:,:,np.newaxis,:],npanel, axis = 2)
    xbar_2d              = np.repeat(np.swapaxes(np.swapaxes(xbar,0, 2),0,1)[:,:,:,np.newaxis],npanel, axis = 3)
    ybar_2d              = np.repeat(np.swapaxes(np.swapaxes(ybar,0, 2),0,1)[:,:,:,np.newaxis],npanel, axis = 3) 
    st_2d                = np.repeat(np.swapaxes(np.swapaxes(st,0, 2),0,1)[:,:,:,np.newaxis],npanel, axis = 3) 
    ct_2d                = np.repeat(np.swapaxes(np.swapaxes(ct,0, 2),0,1)[:,:,:,np.newaxis],npanel, axis = 3) 
    st_2d_T              = np.swapaxes(st_2d,2,3)
    ct_2d_T              = np.swapaxes(ct_2d,2,3)  
    
    # Fill the elements of the matrix of aero influence coefficients
    sti_minus_j          = ct_2d_T*st_2d -  st_2d_T*ct_2d  
    cti_minus_j          = ct_2d_T*ct_2d +  st_2d_T*st_2d
    rij                  = np.sqrt((xbar_2d-x_2d[:,:,:,:-1])**2 + (ybar_2d-y_2d[:,:,:,:-1])**2)
    rij_plus_1           = np.sqrt((xbar_2d-x_2d[:,:,:,1:])**2 +  (ybar_2d-y_2d[:,:,:,1:])**2)
    rij_dot_rij_plus_1   = (xbar_2d-x_2d[:,:,:,:-1])*(xbar_2d-x_2d[:,:,:,1:]) + (ybar_2d-y_2d[:,:,:,:-1])*(ybar_2d-y_2d[:,:,:,1:])  
    anglesign            = np.sign((xbar_2d-x_2d[:,:,:,:-1])*(ybar_2d-y_2d[:,:,:,1:]) - (xbar_2d-x_2d[:,:,:,1:])*(ybar_2d-y_2d[:,:,:,:-1]))
    r_ratio              = rij_dot_rij_plus_1/rij/rij_plus_1
    r_ratio[r_ratio>1.0] = 1.0 # numerical noise 
    betaij               = np.real(anglesign*np.arccos(r_ratio))  
    diag_indices         = list(np.tile(np.repeat(np.arange(npanel),nalpha),nRe))
    aoas                 = list(np.tile(np.arange(nalpha),nRe*npanel))
    res                  = list(np.repeat(np.arange(nRe),nalpha*npanel))   
    betaij[aoas,res,diag_indices,diag_indices] = np.pi 
    
    ainfl[:,:,:-1,:-1]   = pi2inv*(sti_minus_j*np.log(rij_plus_1/rij) + cti_minus_j*betaij)
    mat_1                = np.sum(pi2inv*(cti_minus_j*np.log(rij_plus_1/rij)-sti_minus_j*betaij), axis = 3)
    ainfl[:,:,:-1,-1]    = mat_1  
    
    mat_2                = pi2inv*(sti_minus_j*betaij - cti_minus_j*np.log(rij_plus_1/rij))
    mat_3                = pi2inv*(sti_minus_j*np.log(rij_plus_1/rij) + cti_minus_j*betaij)
    ainfl[:,:,-1,:-1]    = mat_2[:,:,0] + mat_2[:,:,-1]
    ainfl[:,:,-1,-1]     = np.sum(mat_3,axis = 3)[:,:,0] + np.sum(mat_3,axis = 3)[:,:,-1]   
    
    return  ainfl  
