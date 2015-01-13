
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Atmosphere(Analysis):
    """ SUAVE.Analyses.Aerodynamics.Aerodynamics()
    """
    def __defaults__(self):
        self.tag    = 'atmosphere'
        self.features = Data()
        self.settings = Data()
        
        from SUAVE.Attributes.Atmospheres.Earth import US_Standard_1976
        self.features = US_Standard_1976()
        
    def compute_values(self,h):
    
        values = self.features.compute_values(h)
        
        return values
        