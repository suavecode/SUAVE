
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis

# default Aero Results
from Results import Results

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Aerodynamics(Analysis):
    """ SUAVE.Analyses.Aerodynamics.Aerodynamics()
    """
    def __defaults__(self):
        self.tag    = 'aerodynamics'
        
        self.geometry = Data()
        self.settings = Data()
        
        
    def evaluate(self,conditions):
        
        results = Results()
        
        return results
    
    def finalize(self):
        
        return     
    
    __call__ = evaluate
        