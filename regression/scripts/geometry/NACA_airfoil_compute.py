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
    npoints      = 11
    airfoil_data = compute_naca_4series('2410',npoints) 
    
    truth_upper_x  = np.array([0.  , 0.06979572, 0.2563872 , 0.5193338 , 0.80322602,       1.        ])
    truth_lower_x  = np.array([0.  , 0.06979572, 0.2563872 , 0.5193338 , 0.80322602,       1.        ])
    truth_upper_y  = np.array([0.  , 0.04038307, 0.06705705, 0.06227549, 0.03252813,       0.00105   ])
    truth_lower_y  = np.array([0.  , -0.02764178, -0.03221321, -0.02385778, -0.01059382,  -0.00105   ])

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