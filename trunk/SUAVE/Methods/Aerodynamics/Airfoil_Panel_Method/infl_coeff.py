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
def infl_coeff(x,y,xbar,ybar,st,ct,npanel):
    """Compute the matrix of aerodynamic influence  coefficients for later use

    Assumptions:
    None

    Source:
    None 
 
    Inputs
    x       -  Vector of x coordinates of the surface nodes     
    y       -  Vector of y coordinates of the surface nodes      
    xbar    -  x-coordinate of the midpoint of each panel     
    ybar    -  y-coordinate of the midpoint of each panel     
    st      -  np.sin(theta) for each panel                                
    ct      -  np.cos(theta) for each panel                             
    npanel  -  Number of panels on the airfoil                      
                                                                                            
    Outputs                                        
    ainfl   -  Aero influence coefficient matrix    

    Properties Used:
    N/A
    """                         
   
    ainfl                = np.zeros((npanel+1,npanel+1))    
    pi2inv               = 1 / (2*np.pi) 
    
    # convert 1d matrices to 2d 
    x_2d                 = np.repeat(np.atleast_2d(x),npanel, axis = 0) 
    y_2d                 = np.repeat(np.atleast_2d(y),npanel, axis = 0)
    xbar_2d              = np.repeat(np.atleast_2d(xbar).T,npanel, axis = 1)
    ybar_2d              = np.repeat(np.atleast_2d(ybar).T,npanel, axis = 1)
    st_2d                = np.atleast_2d(st)
    ct_2d                = np.atleast_2d(ct) 
    
    # Fill the elements of the matrix of aero influence coefficients
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

    ainfl[:-1,:-1]       = pi2inv*(sti_minus_j*np.log(rij_plus_1/rij) + cti_minus_j*betaij)
    mat_1                = np.sum(pi2inv*(cti_minus_j*np.log(rij_plus_1/rij)-sti_minus_j*betaij), axis = 1)
    ainfl[:-1,-1]        = mat_1  
    
    mat_2                = pi2inv*(sti_minus_j*betaij - cti_minus_j*np.log(rij_plus_1/rij))
    mat_3                = pi2inv*(sti_minus_j*np.log(rij_plus_1/rij) + cti_minus_j*betaij)
    ainfl[-1,:-1]        = mat_2[0] + mat_2[-1]
    ainfl[-1,-1]         = np.sum(mat_3,axis = 1)[0] + np.sum(mat_3,axis = 1)[-1]   
    
    return  ainfl  
