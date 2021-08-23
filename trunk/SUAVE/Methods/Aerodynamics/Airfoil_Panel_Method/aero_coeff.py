## @ingroup Methods-Aerodynamics-Airfoil_Panel_Method
# aero_coeff.py

# Created:  Mar 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
from SUAVE.Core import Data
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
     
    dx      = x[1:]-x[:-1]
    dy      = y[1:]-y[:-1]
    xa      = 0.5*(x[1:] +x[:-1])-0.25
    ya      = 0.5*(y[1:] +y[:-1])
    dcl     = cp[:-1]*dx
    dcd     = -cp[:-1]*dy
    cl_val  = np.sum(dcl,axis=0)
    cd_val  = np.sum(dcd,axis=0)
    cm_val  = np.sum((-dcl*xa + dcd*ya),axis=0) 
    
    cl      = cl_val*np.cos(al) - cd_val*np.sin(al) 
    cd      = cl_val*np.sin(al) + cd_val*np.cos(al)
    cm      = cm_val 
    
    AERO_RES = Data(
        Cl = cl,
        Cd = cd,
        Cm = cm)
    
    return AERO_RES