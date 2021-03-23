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
def infl_coeff(x,y,xbar,ybar,st,ct,ainfl,npanel):
    """Compute the matrix of aerodynamic influence  coefficients for later use

    Assumptions:
    None

    Source:
    None 
 
    Inputs
    x       -  Vector of x coordinates of the surface nodes     
    y       -  Vector of y coordinates of the surface nodes      
    xbar    -  X-coordinate of the midpoint of each panel     
    ybar    -  X-coordinate of the midpoint of each panel     
    st      -  np.sin(theta) for each panel                                
    ct      -  np.cos(theta) for each panel                               
    ainfl   -  Aero influence coefficient matrix                      
    npanel  -  Number of panels on the airfoil                      
                                                                                            
    Outputs                                        
    ainfl   -  Aero influence coefficient matrix   
                       

    Properties Used:
    N/A
    """                         
    
    np.pi2inv = 1 / (2*np.pi)
     
    # Fill the elements of the matrix of aero influence coefficients 
    
    for i_idx in range(npanel): 
        # find contribution of the jth panel 
        for j_idx in range(npanel):
            sti_minus_j = st[i_idx]*ct[j_idx]-ct[i_idx]*st[j_idx]
            cti_minus_j = ct[i_idx]*ct[j_idx]+st[i_idx]*st[j_idx]
            rij=np.sqrt((xbar[i_idx]-x[j_idx])**2 + (ybar[i_idx]-y[j_idx])**2)
            rij_plus_1=np.sqrt((xbar[i_idx]-x[j_idx+1])**2 + (ybar[i_idx]-y[j_idx+1])**2)
            rij_dot_rij_plus_1 = (xbar[i_idx]-x[j_idx])*(xbar[i_idx]-x[j_idx+1]) + (ybar[i_idx]-y[j_idx])*(ybar[i_idx]-y[j_idx+1])
            rij_cross_rij_plus_1 = (xbar[i_idx]-x[j_idx])*(ybar[i_idx]-y[j_idx+1]) - (xbar[i_idx]-x[j_idx+1])*(ybar[i_idx]-y[j_idx]) 
            anglesign = np.sign((xbar[i_idx]-x[j_idx])*(ybar[i_idx]-y[j_idx+1]) - (xbar[i_idx]-x[j_idx+1])*(ybar[i_idx]-y[j_idx]))
            betaij=np.real(anglesign*np.arccos(rij_dot_rij_plus_1/rij/rij_plus_1))
            if i_idx == j_idx:
                betaij = np.pi
            
            ainfl(i_idx,j_idx) = np.pi2inv*(sti_minus_j*np.log(rij_plus_1/rij) + cti_minus_j*betaij)
            ainfl(i_idx,npanel+1) = ainfl(i_idx,npanel+1) + np.pi2inv*(cti_minus_j*np.log(rij_plus_1/rij)-sti_minus_j*betaij)
            if i_idx == 1 or i_idx == npanel:
                ainfl(npanel+1,j_idx) = ainfl(npanel+1,j_idx)+ np.pi2inv*(sti_minus_j*betaij - cti_minus_j*np.log(rij_plus_1/rij))
                ainfl(npanel+1,npanel+1) = ainfl(npanel+1,npanel+1)+np.pi2inv*(sti_minus_j*np.log(rij_plus_1/rij) + cti_minus_j*betaij)
  
    if np.linalg.matrix_rank(ainfl) != npanel+1:
        return  

    return  ainfl  
