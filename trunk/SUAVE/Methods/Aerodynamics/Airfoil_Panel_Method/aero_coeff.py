## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# aero_coeff.py

# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
import numpy as np

# ----------------------------------------------------------------------
# panel_geometry.py
# ----------------------------------------------------------------------   

## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
def aero_coeff(x,y,cp,al,npanel):
    """Compute airfoil force and moment coefficients about 
                    the quarter chord point          

    Assumptions:
    None

    Source:
    None                                                                    
                                                                   
    Inputs:                                                     
    x       -  Vector of x coordinates of the surface nodes      
    y       -  Vector of y coordinates of the surface nodes       
    cp      -  Vector of coefficients of pressure at the nodes  
    al      -  Angle of attack in radians                             
    npanel  -  Number of panels on the airfoil               
                                                                 
    Outputs:                                           
    cl      -  Airfoil lift coefficient                
    cd      -  Airfoil drag coefficient                                 
    cm      -  Airfoil moment coefficient about the c/4     
     
   Properties Used:
    N/A
    """    
    
    cl  = 0
    cd  = 0
    cm  = 0
     
    dx  = x[1:] -x[:-1]
    dy  = y[1:] -y[:-1]
    xa  = 0.5*(x[1:] +x[:-1])-0.25
    ya  = 0.5*(y[1:] +y[:-1])
    dcl = -cp[:-1]*dx
    dcd = cp[:-1]*dy
    cl  = np.sum(dcl)
    cd  = np.sum(dcd)
    cm  = -np.sum(dcd*ya -dcl*xa)
    
    dcl = cl*np.cos(al) -cd*np.sin(al)
    cd  = cl*np.sin(al) +cd*np.cos(al)
    cl  = -dcl  
    
    return cl,cd,cm