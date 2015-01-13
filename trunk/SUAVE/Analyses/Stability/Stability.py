
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results

from SUAVE.Attributes.Flight_Dynamics import Fidelity_Zero


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Stability(Analysis):
    """ SUAVE.Analyses.Stability.Stability()
    """
    def __defaults__(self):
        self.tag    = 'stability'
        self.features = Data()
        self.settings = Data()
        
        self.stability = Fidelity_Zero()
        
        
    def evaluate(self,conditions):
        
        results = self.stability(conditions)
        
        return results
    
    
    def finalize(self):
        
        self.stability.initialize(self.features.vehicle)  
        
        return
    
    
    __call__ = evaluate
        