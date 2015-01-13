
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results
from SUAVE.Methods.Weights.Correlations.Tube_Wing import empty


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
        
        vehicle = self.vehicle
        
        results = empty(vehicle)
        
        return results
    
    __call__ = evaluate
    
    
    def finalize(self):
        
        self.mass_properties = self.vehicle.mass_properties
        
        return
        