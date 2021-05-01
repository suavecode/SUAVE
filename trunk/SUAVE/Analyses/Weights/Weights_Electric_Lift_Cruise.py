## @ingroup Analyses-Weights

# Weights_Electric_Lift_Cruise.py
#
# Created:  Aug 2017, J. Smart
# Modified: Apr 2018, J. Smart
#           Apr 2020, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from .Weights import Weights

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

## @ingroup Analyses-Weights
class Weights_Electric_Lift_Cruise(Weights):
    """ SUAVE.Analyses.Weights.Weights_Electric_Lift_Cruise()
    
    Assumptions:
    None
    
    Source:
    N/A
    
    Inputs:
    N/A
    
    Outputs:
    N/A
    
    Properties Used:
    N/A
    """

    def __defaults__(self):
        """Sets the default parameters for the weight analysis
        Weight method to be used is default Electric Stopped Rotor method.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        N/A

        Outputs:
        N/A

        Properties Used:
        N/A
        """
        
        self.tag = 'weights_electric_stopped_rotor'
        
        self.vehicle  = Data()
        self.settings = Data()
        
        self.settings.empty = SUAVE.Methods.Weights.Buildups.Electric_Lift_Cruise.empty
