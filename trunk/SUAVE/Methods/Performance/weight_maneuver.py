""" weight_maneuver.py: ... """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import math
import copy

from SUAVE.Structure            import Data
from SUAVE.Attributes.Results   import Result, Segment
# from SUAVE.Methods.Utilities    import chebyshev_data, pseudospectral

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def weight_maneuver(maxto):
    """Fuel burned in the warm-up, taxi, take-off, approach, and landing segments. Assumed to be 0.7% of max takeoff weight
        
        Assumptions:
            all segments combined have a fixed fuel burn ratio
        
        Input:
        
        Outputs:
            fuel_burn_maneuver
    """
    # AA 241 Notes Section 11.4
    
    # Calculate
    fuel_burn_maneuever = 0.0035 * maxto # Only calculates all the total Maneuever fuel burn
    
    return fuel_burn_maneuever