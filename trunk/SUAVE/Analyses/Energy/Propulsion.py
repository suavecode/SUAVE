
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Propulsion(Analysis):
    """ SUAVE.Analyses.Energy.Propulsion()
    """
    def __defaults__(self):
        self.tag    = 'propulsion'
        self.vehicle  = Data()
        self.settings = Data()
        
    def evaluate(self,conditions,numerics):
        propulsor=self.propulsor
        F, mdot,P= propulsor.evaluate(conditions, numerics) 
        return F,mdot,P
        