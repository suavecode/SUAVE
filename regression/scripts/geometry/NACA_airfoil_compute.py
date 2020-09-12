# NACA_airfoil_compute.py
# 
# Created:  April 2018, W. Maier
# Modified: Sep 2020, M. Clarke 

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
    airfoil_data = compute_naca_4series(camber, camber_loc, thickness,npoints) 
    
    truth_upper_x  = np.array([0. , 0.08654358, 0.25116155, 0.46508794, 0.71656695,1. ])
    truth_lower_x  = np.array([0. , 0.09234186, 0.25480288, 0.46442806, 0.71451656,1.])
    truth_upper_y  = np.array([0. , 0.04528598, 0.06683575, 0.06561866, 0.04370913,0.])
    truth_lower_y  = np.array([0., -0.02939744, -0.03223931, -0.02608462, -0.01477209, 0.])

    # Compute Errors
    error       = Data() 
    error.upper_x = np.abs( airfoil_data.x_upper_surface[0] - truth_upper_x)
    error.lower_x = np.abs( airfoil_data.x_lower_surface[0] - truth_lower_x)
    error.upper_y = np.abs( airfoil_data.y_upper_surface[0] - truth_upper_y)
    error.lower_y = np.abs( airfoil_data.y_lower_surface[0] - truth_lower_y)    
    
    for k,v in list(error.items()):
        assert np.any(np.abs(v)<1e-6)
    
if __name__ == '__main__':
    
    main()
    
    print('NACA regression test passed!')   