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
def append_vortex_rings(wVD, VD_TE_Rings): 
    """
    Appends a new row of ring vortices shed from trailing edge to the previous set of wake vortex distributions.
    
    Inputs:
       wVD         - wake vortex distribution
       VD_TE_Rings - new set of vortex rings to append to wake vortex distribution
       
    Outputs:
       wVD         - new total wake vortex distribution
    
    """
    if np.size(wVD.XA1) ==0:
        # first pass
        wVD.XA1 = VD_TE_Rings.XA1
        wVD.XA2 = VD_TE_Rings.XA2
        wVD.XB1 = VD_TE_Rings.XB1
        wVD.XB2 = VD_TE_Rings.XB2
        
        wVD.YA1 = VD_TE_Rings.YA1
        wVD.YA2 = VD_TE_Rings.YA2
        wVD.YB1 = VD_TE_Rings.YB1
        wVD.YB2 = VD_TE_Rings.YB2
        
        wVD.ZA1 = VD_TE_Rings.ZA1
        wVD.ZA2 = VD_TE_Rings.ZA2
        wVD.ZB1 = VD_TE_Rings.ZB1
        wVD.ZB2 = VD_TE_Rings.ZB2

        wVD.XC = VD_TE_Rings.XC
        wVD.YC = VD_TE_Rings.YC
        wVD.ZC = VD_TE_Rings.ZC         
        
        wVD.GAMMA = VD_TE_Rings.GAMMA
    else:
        wVD.XA1 = np.atleast_2d(np.append(wVD.XA1, VD_TE_Rings.XA1))
        wVD.XA2 = np.atleast_2d(np.append(wVD.XA2, VD_TE_Rings.XA2))
        wVD.XB1 = np.atleast_2d(np.append(wVD.XB1, VD_TE_Rings.XB1))
        wVD.XB2 = np.atleast_2d(np.append(wVD.XB2, VD_TE_Rings.XB2))
        
        wVD.YA1 = np.atleast_2d(np.append(wVD.YA1, VD_TE_Rings.YA1))
        wVD.YA2 = np.atleast_2d(np.append(wVD.YA2, VD_TE_Rings.YA2))
        wVD.YB1 = np.atleast_2d(np.append(wVD.YB1, VD_TE_Rings.YB1))
        wVD.YB2 = np.atleast_2d(np.append(wVD.YB2, VD_TE_Rings.YB2))
    
        wVD.ZA1 = np.atleast_2d(np.append(wVD.ZA1, VD_TE_Rings.ZA1))
        wVD.ZA2 = np.atleast_2d(np.append(wVD.ZA2, VD_TE_Rings.ZA2))
        wVD.ZB1 = np.atleast_2d(np.append(wVD.ZB1, VD_TE_Rings.ZB1))
        wVD.ZB2 = np.atleast_2d(np.append(wVD.ZB2, VD_TE_Rings.ZB2))   

        wVD.XC = np.atleast_2d(np.append(wVD.XC, VD_TE_Rings.XC))
        wVD.YC = np.atleast_2d(np.append(wVD.YC, VD_TE_Rings.YC))
        wVD.ZC = np.atleast_2d(np.append(wVD.ZC, VD_TE_Rings.ZC))        
    
        wVD.GAMMA = np.atleast_2d(np.append(wVD.GAMMA, VD_TE_Rings.GAMMA))
    
    
    wVD.n_cp = len(wVD.XC)
    
    # Compress Data into 1D Arrays  
    mat6_size = (1,np.size(wVD.XA1)) 
    
    wVD_collapsed = Data()
    wVD_collapsed.XA1    =  np.reshape(wVD.XA1,mat6_size)
    wVD_collapsed.YA1    =  np.reshape(wVD.YA1,mat6_size)
    wVD_collapsed.ZA1    =  np.reshape(wVD.ZA1,mat6_size)
    wVD_collapsed.XA2    =  np.reshape(wVD.XA2,mat6_size)
    wVD_collapsed.YA2    =  np.reshape(wVD.YA2,mat6_size)
    wVD_collapsed.ZA2    =  np.reshape(wVD.ZA2,mat6_size)
    wVD_collapsed.XB1    =  np.reshape(wVD.XB1,mat6_size)
    wVD_collapsed.YB1    =  np.reshape(wVD.YB1,mat6_size)
    wVD_collapsed.ZB1    =  np.reshape(wVD.ZB1,mat6_size)
    wVD_collapsed.XB2    =  np.reshape(wVD.XB2,mat6_size)
    wVD_collapsed.YB2    =  np.reshape(wVD.YB2,mat6_size)
    wVD_collapsed.ZB2    =  np.reshape(wVD.ZB2,mat6_size)

    wVD_collapsed.XC    =  np.reshape(wVD.XC,mat6_size)
    wVD_collapsed.YC    =  np.reshape(wVD.YC,mat6_size)
    wVD_collapsed.ZC    =  np.reshape(wVD.ZC,mat6_size)    
    
    wVD_collapsed.GAMMA  =  np.reshape(wVD.GAMMA,mat6_size)    

    wVD_collapsed.n_cp = wVD.n_cp
    
    return wVD, wVD_collapsed