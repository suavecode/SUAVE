## @ingroup Analyses-Weights
# Weights_Tube_Wing.py
#
# Created: Apr 2017, Matthew Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from Weights import Weights


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

## @ingroup Analyses-Weights
class Weights_Tube_Wing(Weights):
    """ This is class that evaluates the weight of Tube and Wing aircraft
    
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
        self.tag = 'weights_tube_wing'
        
        self.vehicle  = Data()
        self.settings = Data()
        
        
    def evaluate(self,conditions=None):
        
        # unpack
        vehicle = self.vehicle
        empty   = SUAVE.Methods.Weights.Correlations.Tube_Wing.empty

        
        # evaluate
        results = empty(vehicle)
        
        # storing weigth breakdown into vehicle
        vehicle.weight_breakdown = results 

        # updating empty weight
        vehicle.mass_properties.operating_empty = results.empty
              
        # done!
        return results
    
    
    def finalize(self):
        
        self.mass_properties = self.vehicle.mass_properties
        
        return
        
