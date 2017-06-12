# Weights.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff
# Modified: Apr 2017, Matthew Clarke

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
        
        
    def evaluate(self,conditions=None):
        
        # unpack
        vehicle = self.vehicle
        
        if vehicle.fuselages.keys() == []:
            empty   = SUAVE.Methods.Weights.Correlations.UAV.empty     #UAV correlations are for flying wing. Need to correct    
        elif vehicle.fuselages.has_key('fuselage'):
            empty   = SUAVE.Methods. Weights.Correlations.Tube_Wing.empty
        elif vehicle.fuselages.has_key('fuselage_bwb'):
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
        