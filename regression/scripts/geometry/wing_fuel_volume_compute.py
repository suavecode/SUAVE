# wing_fuel_volume_compute.py
# 
# Created:  April 2018, W. Maier
# Modified: 

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data

import numpy as np

from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_fuel_volume

# ---------------------------------------------------------------------- 
#   Main
# ----------------------------------------------------------------------
def main():
    
    # ------------------------------------------------------------------
    # Test wing fuel volume 
    # ------------------------------------------------------------------
    
    # Setup
    wing      = Data()
    wing.areas = Data()
    
    wing.areas.reference    = 1400.0
    wing.aspect_ratio       = 6.0
    wing.thickness_to_chord = 0.08 

    # Calculation
    wing_fuel_volume(wing)
    
    # Set Truth
    truth_volume = 846.8599884278394 #[m^2]
    
    # Compute Errors
    error        = Data() 
    error.volume = np.abs(wing.fuel_volume-truth_volume)/truth_volume

    for k,v in list(error.items()):
        assert np.any(np.abs(v)<1e-6)
    
if __name__ == '__main__':
    
    main()  
    print('Fuel volume regression test passed!')   