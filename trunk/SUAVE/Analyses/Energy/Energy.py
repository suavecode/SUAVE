
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
        
        
    def evaluate_thrust(self,conditions):
        network=self.network
        F, mdot= network.evaluate_thrust(conditions, numerics) 
        return Results()
    