## @ingroup Analyses-Weights
# Weights_UAV.py
#
# Created:  Apr 2017, Matthew Clarke
# Modified: Apr 2020, E. Botero

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
class Weights_UAV(Weights):
    """ This is class that evaluates the weight of a UAV
    
    Assumptions:
        None

    Source:
        N/A

    Inputs:
        None
      
    Outputs:
        None

    Properties Used:
        N/A
    """
    def __defaults__(self):
        """This sets the default values and methods for the UAV weight analysis.
    
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """           
        self.tag = 'weights_uav'
        
        self.vehicle  = Data()
        self.settings = Data()
        
        self.settings.empty = SUAVE.Methods.Weights.Correlations.UAV.empty