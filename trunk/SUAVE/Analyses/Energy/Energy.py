
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
        
    def evaluate(self,state):
        network    = self.network
        results = network.evaluate_thrust(state) 
        
        return results
    