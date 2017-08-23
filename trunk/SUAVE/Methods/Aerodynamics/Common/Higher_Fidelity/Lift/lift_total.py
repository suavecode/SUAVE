## @ingroup Methods-Aerodynamics-Higher_Fidelity-Lift
# lift_total.py
# 
# Created:  Feb 2016, E. Botero
# Modified: Jun 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data, Units


# ----------------------------------------------------------------------
#  Lift Total
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Higher_Fidelity-Lift
def lift_total(state,settings,geometry):
    """Extract the lift coefficient

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    state.conditions.aerodynamics.lift_coefficient [Unitless]

    Outputs:
    CL (coefficient of lift)                       [Unitless]

    Properties Used:
    N/A
    """  
    
    CL = state.conditions.aerodynamics.lift_coefficient     

    return CL