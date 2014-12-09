
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results

from SUAVE.Attributes.Flight_Dynamics.Fidelity_Zero import Fidelity_Zero as stability


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
        
        self.stability = stability()
        
        
    def evaluate(self,conditions):
        
        results = self.stability(conditions)
        
        return results
    
    
    def finalize(self):
        
        self.stability.initialize(self.features)  
        
        return
    
    
    __call__ = evaluate
        