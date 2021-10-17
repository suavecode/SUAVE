## @ingroup Methods-Aerodynamics-Common-Fidelity_One-Free_Wake
#  free_wake_trailing_edge_rings.py
# 
# Created:  Oct 2021, R. ERhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from SUAVE.Core import Data 
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.compute_wake_contraction_matrix import compute_wake_contraction_matrix
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry   

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift   
def append_vortex_rings(wVD_collapsed, VD_TE_collapsed): 
    """
    Appends a new row of ring vortices shed from trailing edge to the previous set of wake vortex distributions.
    
    Inputs:
       wVD         - wake vortex distribution
       VD_TE_Rings - new set of vortex rings to append to wake vortex distribution
       
    Outputs:
       wVD         - new total wake vortex distribution
    
    """
    if np.size(wVD_collapsed.XA1) ==0:
        # first pass
        wVD_collapsed.XA1 = VD_TE_collapsed.XA1
        wVD_collapsed.XA2 = VD_TE_collapsed.XA2
        wVD_collapsed.XB1 = VD_TE_collapsed.XB1
        wVD_collapsed.XB2 = VD_TE_collapsed.XB2
        
        wVD_collapsed.YA1 = VD_TE_collapsed.YA1
        wVD_collapsed.YA2 = VD_TE_collapsed.YA2
        wVD_collapsed.YB1 = VD_TE_collapsed.YB1
        wVD_collapsed.YB2 = VD_TE_collapsed.YB2
        
        wVD_collapsed.ZA1 = VD_TE_collapsed.ZA1
        wVD_collapsed.ZA2 = VD_TE_collapsed.ZA2
        wVD_collapsed.ZB1 = VD_TE_collapsed.ZB1
        wVD_collapsed.ZB2 = VD_TE_collapsed.ZB2
        
        wVD_collapsed.XC = VD_TE_collapsed.XC
        wVD_collapsed.YC = VD_TE_collapsed.YC
        wVD_collapsed.ZC = VD_TE_collapsed.ZC         
        
        wVD_collapsed.GAMMA = VD_TE_collapsed.GAMMA
    else:
        wVD_collapsed.XA1 = np.atleast_2d(np.append(wVD_collapsed.XA1, VD_TE_collapsed.XA1))
        wVD_collapsed.XA2 = np.atleast_2d(np.append(wVD_collapsed.XA2, VD_TE_collapsed.XA2))
        wVD_collapsed.XB1 = np.atleast_2d(np.append(wVD_collapsed.XB1, VD_TE_collapsed.XB1))
        wVD_collapsed.XB2 = np.atleast_2d(np.append(wVD_collapsed.XB2, VD_TE_collapsed.XB2))
        
        wVD_collapsed.YA1 = np.atleast_2d(np.append(wVD_collapsed.YA1, VD_TE_collapsed.YA1))
        wVD_collapsed.YA2 = np.atleast_2d(np.append(wVD_collapsed.YA2, VD_TE_collapsed.YA2))
        wVD_collapsed.YB1 = np.atleast_2d(np.append(wVD_collapsed.YB1, VD_TE_collapsed.YB1))
        wVD_collapsed.YB2 = np.atleast_2d(np.append(wVD_collapsed.YB2, VD_TE_collapsed.YB2))
        
        wVD_collapsed.ZA1 = np.atleast_2d(np.append(wVD_collapsed.ZA1, VD_TE_collapsed.ZA1))
        wVD_collapsed.ZA2 = np.atleast_2d(np.append(wVD_collapsed.ZA2, VD_TE_collapsed.ZA2))
        wVD_collapsed.ZB1 = np.atleast_2d(np.append(wVD_collapsed.ZB1, VD_TE_collapsed.ZB1))
        wVD_collapsed.ZB2 = np.atleast_2d(np.append(wVD_collapsed.ZB2, VD_TE_collapsed.ZB2))  
        
        wVD_collapsed.XC = np.atleast_2d(np.append(wVD_collapsed.XC, VD_TE_collapsed.XC))
        wVD_collapsed.YC = np.atleast_2d(np.append(wVD_collapsed.YC, VD_TE_collapsed.YC))
        wVD_collapsed.ZC = np.atleast_2d(np.append(wVD_collapsed.ZC, VD_TE_collapsed.ZC))     
        
        wVD_collapsed.GAMMA = np.atleast_2d(np.append(wVD_collapsed.GAMMA, VD_TE_collapsed.GAMMA))
    
    
    wVD_collapsed.n_cp = len(wVD_collapsed.XC)
    
    ## Compress Data into 1D Arrays  
    #mat6_size = (1,np.size(wVD_collapsed.XA1)) 
    
    #wVD_collapsed = Data()
    #wVD_collapsed.XA1    =  np.reshape(wVD_collapsed.XA1,mat6_size)
    #wVD_collapsed.YA1    =  np.reshape(wVD_collapsed.YA1,mat6_size)
    #wVD_collapsed.ZA1    =  np.reshape(wVD_collapsed.ZA1,mat6_size)
    #wVD_collapsed.XA2    =  np.reshape(wVD_collapsed.XA2,mat6_size)
    #wVD_collapsed.YA2    =  np.reshape(wVD_collapsed.YA2,mat6_size)
    #wVD_collapsed.ZA2    =  np.reshape(wVD_collapsed.ZA2,mat6_size)
    #wVD_collapsed.XB1    =  np.reshape(wVD_collapsed.XB1,mat6_size)
    #wVD_collapsed.YB1    =  np.reshape(wVD_collapsed.YB1,mat6_size)
    #wVD_collapsed.ZB1    =  np.reshape(wVD_collapsed.ZB1,mat6_size)
    #wVD_collapsed.XB2    =  np.reshape(wVD_collapsed.XB2,mat6_size)
    #wVD_collapsed.YB2    =  np.reshape(wVD_collapsed.YB2,mat6_size)
    #wVD_collapsed.ZB2    =  np.reshape(wVD_collapsed.ZB2,mat6_size)

    #wVD_collapsed.XC    =  np.reshape(wVD_collapsed.XC,mat6_size)
    #wVD_collapsed.YC    =  np.reshape(wVD_collapsed.YC,mat6_size)
    #wVD_collapsed.ZC    =  np.reshape(wVD_collapsed.ZC,mat6_size)    
    
    #wVD_collapsed.GAMMA  =  np.reshape(wVD_collapsed.GAMMA,mat6_size)    

    #wVD_collapsed.n_cp = wVD_collapsed.n_cp
    
    return wVD_collapsed