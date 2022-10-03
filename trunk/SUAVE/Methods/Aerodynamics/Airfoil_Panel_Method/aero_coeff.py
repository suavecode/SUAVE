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
    dcn     = cp[:-1]*dx
    dca     = -cp[:-1]*dy
    
    # compute differential forces
    cn      = np.sum(dcn,axis=0)
    ca      = np.sum(dca,axis=0)
    cm      = np.sum((-dcn*xa + dca*ya),axis=0) 
    
    # orient normal and axial forces 
    cl      = cn*np.cos(al) - ca*np.sin(al) 
    cd      = cn*np.sin(al) + ca*np.cos(al) 
    
    # pack results
    AERO_RES = Data(
        Cl = cl,
        Cd = cd,
        Cm = cm)
    
    return AERO_RES