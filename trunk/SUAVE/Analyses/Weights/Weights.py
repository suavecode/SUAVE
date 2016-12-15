# Weights.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff
# Modified: Aug 2016, T. Orra, D. Bianchi

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from SUAVE.Analyses import Analysis


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Weights(Analysis):
    """ SUAVE.Analyses.Weights.Weights()
    """
    def __defaults__(self):
        self.tag = 'weights'
        self.vehicle  = Data()
        
        self.settings = Data()
        self.settings.empty_weight_method = \
            SUAVE.Methods.Weights.Correlations.Tube_Wing.empty
        self.settings.empty_weight_increment = 0.

    def evaluate(self,conditions=None):
        
        # unpack
        vehicle = self.vehicle
        empty   = self.settings.empty_weight_method
        
        # evaluate
        results = empty(vehicle)

        # applying empty weight offset
        results.empty += self.settings.empty_weight_increment

        # storing weigth breakdown into vehicle
        vehicle.weight_breakdown = results 

        # updating empty weight
        vehicle.mass_properties.operating_empty = results.empty
        
        # done!
        return results
    
    
    def finalize(self):
        
        self.mass_properties = self.vehicle.mass_properties
        
        return
        