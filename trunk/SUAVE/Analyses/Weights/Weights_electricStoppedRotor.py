## @ingroup Analyses-Weights

# Weights_electricStoppedRotor.py
#
# Created: Aug 2017, J. Smart
# Modified: Mar 2018, J. Smart

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from Weights import Weights


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Weights_electricStoppedRotor(Weights):
    """ SUAVE.Analyses.Weights.Weights_electricStoppedRotor()
    """
    def __defaults__(self):
        self.tag = 'weights_electric_stopped_rotor'
        
        self.vehicle  = Data()
        self.settings = Data()
        
        
    def evaluate(self,conditions=None):
        
        # unpack
        vehicle = self.vehicle
        empty   = SUAVE.Methods.Weights.Buildups.electricStoppedRotor.empty

        
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
        
