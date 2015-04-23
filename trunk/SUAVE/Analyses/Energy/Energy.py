
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Energy(Analysis):
    """ SUAVE.Analyses.Energy.Energy()
    """
    def __defaults__(self):
        self.tag    = 'energy'
        self.features = Data()
        self.settings = Data()
        
        
    def evaluate(self,conditions, numerics):
        network=self.network
        F, mdot, P= network.evaluate_thrust(conditions, numerics) 
        return F, mdot, P
    