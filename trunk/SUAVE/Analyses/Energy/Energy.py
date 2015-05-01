
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
        self.tag     = 'energy'
        self.network = None
        
    def evaluate_thrust(self,state):
        network = self.network
        results = network.evaluate_thrust(state) 
        
        return results
    