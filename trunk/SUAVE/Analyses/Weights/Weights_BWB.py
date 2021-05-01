## @ingroup Analyses-Weights 
# Weights_BWB.py
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
class Weights_BWB(Weights):
    """ This is class that evaluates the weight of a BWB aircraft
    
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
        """This sets the default values and methods for the BWB weight analysis.
    
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
        self.tag = 'weights_bwb'
        
        self.vehicle  = Data()
        self.settings = Data()
        
        self.settings.empty = SUAVE.Methods.Weights.Correlations.BWB.empty