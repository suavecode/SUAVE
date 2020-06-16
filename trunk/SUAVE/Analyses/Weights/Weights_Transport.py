## @ingroup Analyses-Weights
# Weights_Transport.py
#
# Created:  Apr 2017, Matthew Clarke
# Modified: Oct 2017, T. MacDonald
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
class Weights_Transport(Weights):
    """ This is class that evaluates the weight of Transport class aircraft
    
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
        """This sets the default values and methods for the tube and wing 
        aircraft weight analysis.

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
        self.tag = 'weights_transport'
        
        self.vehicle  = Data()
        self.settings = Data()
        self.settings.weight_reduction_factors = Data()
        
        # Reduction factors are proportional (.1 is a 10% weight reduction)
        self.settings.weight_reduction_factors.main_wing = 0.
        self.settings.weight_reduction_factors.fuselage  = 0.
        self.settings.weight_reduction_factors.empennage = 0. # applied to horizontal and vertical stabilizers
        
        self.settings.empty = SUAVE.Methods.Weights.Correlations.Transport.empty