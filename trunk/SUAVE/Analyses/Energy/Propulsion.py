
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Propulsion(Analysis):
    """ SUAVE.Analyses.Energy.Propulsion()
    """
    def __defaults__(self):
        self.tag    = 'propulsion'
        self.features = Data()
        self.settings = Data()
        
        
    def evaluate(self,conditions,numerics):
        
        network  = self.features.propulsors
        
        F,mdot,P = network.evaluate(conditions,numerics)
        
        return F,mdot,P
    
    __call__ = evaluate
        