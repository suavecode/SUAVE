
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results


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
        
        
    def evaluate(self,condtitions):
        return Results()
    
    __call__ = evaluate
        