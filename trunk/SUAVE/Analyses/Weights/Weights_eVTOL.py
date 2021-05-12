## @ingroup Analyses-Weights

# Weights_eVTOL.py
#
# Created:  Aug, 2017, J. Smart
# Modified: Apr, 2018, J. Smart
#           May, 2021, M. Clarke

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
class Weights_eVTOL(Weights):
    """This is class that evaluates the weight of an eVTOL aircraft
    
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
        """Sets the default parameters for an eVTOL weight analysis

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
        
        self.tag = 'weights_evtol'
        
        self.vehicle  = Data()
        self.settings = Data()
        
        self.settings.empty = SUAVE.Methods.Weights.Buildups.eVTOL.empty