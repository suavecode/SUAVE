""" Atmospheric.py: The base atmosphere analysis class."""
## @ingroup Atmospheric
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

## @ingroup Atmospheric
class Atmospheric(Analysis):
    """ This is the base class for atmospheric Analyses
    """
    def __defaults__(self):
        """ This sets the default values for the analysis to function.
        Inputs:
        Base atmosphere attribute class
    
            """          
        atmo_data = Atmosphere()
        self.update(atmo_data)
        
        
    def compute_values(self,altitude):
        raise NotImplementedError
