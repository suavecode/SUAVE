#
#
# Modified  2/16/15, Tim MacDonald
# Changing atmospheric analysis format


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes.Atmospheres.Atmosphere import Atmosphere
from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Analyses import Analysis, Results


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Atmospheric(Analysis):
    """ SUAVE.Analyses.Aerodynamics.Aerodynamics()
    """
    def __defaults__(self):
        atmo_data = Atmosphere()
        self.update(atmo_data)
        
        
    def compute_values(self,altitude):
        raise NotImplementedError
