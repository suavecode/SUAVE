
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results
from SUAVE.Methods.Weights.Correlations.Tube_Wing import empty


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Weights(Analysis):
    """ SUAVE.Analyses.Weights.Weights()
    """
    def __defaults__(self):
        self.tag    = 'weights'
        self.features = Data()
        self.settings = Data()
        
        
    def evaluate(self,conditions):
        
        results = empty(conditions)
        
        return results
    
    __call__ = evaluate
        