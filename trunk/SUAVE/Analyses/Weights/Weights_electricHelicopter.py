## @ ingroup Analyses-Weights

# Weights_electricHelicopter.py
#
# Created: Mar 2017, J. Smart

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from Weights import Weights


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Weights_electricHelicopter(Weights):
    """ SUAVE.Analyses.Weights.Weights_electricHelicopter()
    """
    def __defaults__(self):
        self.tag = 'weights_electric_helicopter'
        
        self.vehicle  = Data()
        self.settings = Data()
        
        
    def evaluate(self,conditions=None):
        
        # unpack
        vehicle = self.vehicle
        empty   = SUAVE.Methods.Weights.Buildups.electricHelicopter.empty

        
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
        
