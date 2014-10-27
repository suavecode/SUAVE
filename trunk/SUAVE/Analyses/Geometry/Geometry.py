
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Geometry(Analysis):
    """ SUAVE.Analyses.Geometry.Geometry()
    """
    def __defaults__(self):
        self.tag    = 'geometry'
        self.features = Data()
        self.settings = Data()
        
        
    def evaluate(self,condtitions):
        return Results()
    
    __call__ = evaluate
        