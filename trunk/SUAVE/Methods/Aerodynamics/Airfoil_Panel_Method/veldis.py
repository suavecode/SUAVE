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
    vt = np.zeros(npanel)
    gamma = qg[npanel+1]
    for i_idx in range(npanel):
        vt[i_idx] = ct[i_idx]*np.cos(al) +	 st[i_idx]*np.sin(al)
        for j_idx in range(npanel):
            sti_minus_j = st[i_idx]*ct[j_idx]-ct[i_idx]*st[j_idx]
            cti_minus_j = ct[i_idx]*ct[j_idx]+st[i_idx]*st[j_idx]
            rij=np.sqrt((xbar[i_idx]-x[j_idx])^2 + (ybar[i_idx]-y[j_idx])^2)
            rij_plus_1=np.sqrt((xbar[i_idx]-x[j_idx +1])^2 + (ybar[i_idx]-y[j_idx +1])^2)
            rij_dot_rij_plus_1 = (xbar[i_idx]-x[j_idx])*(xbar[i_idx]-x[j_idx +1]) + (ybar[i_idx]-y[j_idx])*(ybar[i_idx]-y[j_idx +1]) 
            anglesign = np.sign((xbar[i_idx]-x[j_idx])*(ybar[i_idx]-y[j_idx +1]) - (xbar[i_idx]-x[j_idx +1])*(ybar[i_idx]-y[j_idx]))
            betaij= np.real(anglesign*np.acos(rij_dot_rij_plus_1/rij/rij_plus_1))
            if i_idx == j_idx:
                betaij = np.pi

            vt[i_idx] = vt[i_idx] + qg[j_idx]/2/np.pi*(sti_minus_j*betaij - cti_minus_j*np.np.log(rij_plus_1/rij))  + gamma/2/np.pi*(sti_minus_j*np.np.log(rij_plus_1/rij) + cti_minus_j*betaij)

    return  vt  
