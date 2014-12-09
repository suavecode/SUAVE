
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results
from SUAVE.Attributes.Atmospheres import Atmosphere
from SUAVE.Attributes.Atmospheres.Earth import US_Standard_1976
from SUAVE.Attributes.Planets import Earth

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Atmospheres(Analysis):
    """ SUAVE.Analyses.Aerodynamics.Aerodynamics()
    """
    def __defaults__(self):
        self.tag    = 'atmospheres'
        self.features = Data()
        self.settings = Data()
        
        self.atmosphere = US_Standard_1976()
        
    def compute_values(self,conditions):
    
        values = self.atmosphere.compute_values(conditions)
        
        return values
        