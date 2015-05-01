
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results


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
        
        
    def evaluate(self,conditions=None):
        
        # unpack
        vehicle = self.vehicle
        empty   = self.settings.empty_weight_method
        
        # evaluate
        results = empty(vehicle)
        
        # done!
        return results
    
    
    def finalize(self):
        
        self.mass_properties = self.vehicle.mass_properties
        
        return
        