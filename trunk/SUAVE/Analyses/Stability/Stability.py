
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Stability(Analysis):
    """ SUAVE.Analyses.Stability.Stability()
    """
    
    def __defaults__(self):
        self.tag    = 'stability'
        self.geometry = Data()
        self.settings = Data()
        
    def evaluate(self,conditions):
        
        results = Results()
        
        return results
    
    
    def finalize(self):
        
        return
    
    
        