## @ingroup Analyses-Weights

# Weights_Electric_Multicopter.py
#
# Created:  Mar 2017, J. Smart
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
class Weights_Electric_Multicopter(Weights):
    """ SUAVE.Analyses.Weights.Weights_Electric_Multicopter()
    
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
        Weight method to be used is default Electric Multicopter method.
        
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
        
        self.tag = 'weights_electric_helicopter'
        
        self.vehicle  = Data()
        self.settings = Data()
        
        self.settings.empty = SUAVE.Methods.Weights.Buildups.Electric_Multicopter.empty