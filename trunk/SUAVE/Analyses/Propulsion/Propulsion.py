
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Propulsion(Analysis):
    """ SUAVE.Analyses.Propulsion.Propulsion()
    """
    def __defaults__(self):
        self.tag    = 'propulsion'
        self.features = Data()
        self.settings = Data()
        
        
    def evaluate(self,condtitions):
        return Results()
    
    __call__ = evaluate
        