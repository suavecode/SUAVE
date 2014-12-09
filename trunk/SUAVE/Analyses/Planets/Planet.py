
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results
from SUAVE.Attributes.Planets.Planet import Planet
from SUAVE.Attributes.Planets.Earth import Earth

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Planets(Analysis):
    """ SUAVE.Analyses.Aerodynamics.Aerodynamics()
    """
    def __defaults__(self):
        self.tag    = 'planets'
        self.features = Data()
        self.settings = Data()
        
        self.planet = Earth()
        
        
    def gravity(self,conditions):
        
        results = self.sea_level_gravity
        
        return results
        