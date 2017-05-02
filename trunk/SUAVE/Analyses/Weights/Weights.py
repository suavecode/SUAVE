# Weights.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

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
        
        #self.settings.empty_weight_method = \
            #SUAVE.Methods.Weights.Correlations.Tube_Wing.empty
        
        
    def evaluate(self,conditions=None):
        
        # unpack
        vehicle = self.vehicle
        
        if vehicle.configuration == 'Tube_Wing':
            empty   = SUAVE.Methods.Weights.Correlations.Tube_Wing.empty
        elif vehicle.configuration == 'BWB':
            empty   = SUAVE.Methods.Weights.Correlations.BWB.empty
        
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
        