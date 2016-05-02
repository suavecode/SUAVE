# Stability.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
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
    
    
        