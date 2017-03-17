## @ingroup analyses-atmospheric
# Atmospheric.py
#
# Created:  Feb 2015, T. MacDonald
# Modified: Feb 2016, A. Wendorff


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Attributes.Atmospheres.Atmosphere import Atmosphere
from SUAVE.Analyses import Analysis


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

## @ingroup analyses-atmospheric
class Atmospheric(Analysis):
    """ This is the base class for atmospheric analyses. It contains functions
    that are build the default class.
    
    Assumptions:
    None
    
    Source:
    N/A
    """
    def __defaults__(self):
        """This sets the default values for the analysis to function. (I don't actually
        understand what's happening here). Sets the class atmosphere attribute.
        
        Assumptions:
        None
        
        Source:
        N/A
        
        Inputs:
        None
        
        Outputs:
        None
        
        Properties Used:
        None.
            """          
        atmo_data = Atmosphere()
        self.update(atmo_data)
        
        
    def compute_values(self,altitude):
        """This function is not implemented."""
        raise NotImplementedError
