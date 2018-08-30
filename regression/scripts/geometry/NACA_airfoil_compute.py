# NACA_airfoil_compute.py
# 
# Created:  April 2018, W. Maier
# Modified: 

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data

import numpy as np

from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil import compute_naca_4series

# ---------------------------------------------------------------------- 
#   Main
# ----------------------------------------------------------------------
def main():
    
    # ------------------------------------------------------------------
    # Testing
    # Using NACA 2410
    # ------------------------------------------------------------------
    camber       = 0.02
    camber_loc   = 0.4
    thickness    = 0.10
    npoints      = 10
    upper,lower  = compute_naca_4series(camber, camber_loc, thickness,npoints) 
    
    truth_upper      = ([[ 0.        ,  0.        ],
                         [ 0.08654358,  0.04528598],
                         [ 0.25116155,  0.06683575],
                         [ 0.46508794,  0.06561866],
                         [ 0.71656695,  0.04370913],
                         [ 1.        ,  0.        ]])
    
    truth_lower      = ([[ 0.        ,  0.        ],
                         [ 0.09234186, -0.02939744],
                         [ 0.25480288, -0.03223931],
                         [ 0.46442806, -0.02608462],
                         [ 0.71451656, -0.01477209],
                         [ 1.        ,  0.        ]])
    

    # Compute Errors
    error       = Data() 
    error.upper = np.abs(upper-truth_upper)
    error.lower = np.abs(lower-truth_lower)
    
    for k,v in list(error.items()):
        assert np.any(np.abs(v)<1e-6)
    
if __name__ == '__main__':
    
    main()
    
    print('NACA regression test passed!')   