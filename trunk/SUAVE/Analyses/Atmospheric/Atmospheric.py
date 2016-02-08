# Atmospheric.py
#
# Modified  2/16/15, Tim MacDonald
# Modified: Feb 2016, Andrew Wendorff


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes.Atmospheres.Atmosphere import Atmosphere
from SUAVE.Analyses import Analysis


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
