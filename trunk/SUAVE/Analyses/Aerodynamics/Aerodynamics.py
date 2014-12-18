
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results
from SUAVE.Attributes.Aerodynamics import Fidelity_Zero

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Aerodynamics(Analysis):
    """ SUAVE.Analyses.Aerodynamics.Aerodynamics()
    """
    def __defaults__(self):
        self.tag    = 'aerodynamics'
        self.features = Data()
        self.settings = Data()
        
        self.old_aero = Fidelity_Zero()
        
        
    def evaluate(self,conditions):
        
        
        results = self.old_aero(conditions)
        
        return results
        #return Results()
    
    def finalize(self):
        
        self.old_aero.initialize(self.features.vehicle)        
        return     
    
    __call__ = evaluate
        