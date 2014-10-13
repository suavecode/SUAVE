""" fuel_reserve.py: ... """

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

def fuel_reserve(mission,maxto,mzfw_ratio,fuel_cruise):
    """ Setting a prescribed reserve fuel value of 8% of MZFW from Pass Notes
        
        Inputs:
            mission.segment['Reserve'].fFuelRes
        
        Outputs:
            reserve_fuel
        
    """
    #AA 241 Notes Section 11.4
    if mission[0].seg_type == 'pass':
        mzfw = mzfw_ratio * maxto
        reserve_fuel = 0.08 * mzfw
    else:
        # Unpack
        frac_burn = mission.segment['Reserve'].fFuelRes
        reserve_fuel = frac_burn * fuel_cruise

    return reserve_fuel
