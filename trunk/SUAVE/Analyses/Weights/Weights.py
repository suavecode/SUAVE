## @ingroup Analyses-Weights
# Weights.py
#
# Created: Apr 2017, Matthew Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from SUAVE.Analyses import Analysis



# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

## @ingroup Analyses-Weights
class Weights(Analysis):
    """ This is a class that call the functions that computes the weifht of 
    an aircraft depending on its configration
    
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
        self.tag = 'weights'
        
        self.vehicle  = Data()
        self.settings = Data()
               
        
    def evaluate(self):
        
        return 
    
    def finalize(self):
        
        return     
    
        