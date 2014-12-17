
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Planet(Analysis):
    """ SUAVE.Analyses.Aerodynamics.Aerodynamics()
    """
    def __defaults__(self):
        self.tag    = 'planet'
        self.features = Data()
        self.settings = Data()
        
        from SUAVE.Attributes.Planets.Earth import Earth
        self.features = Earth()
        
        
        