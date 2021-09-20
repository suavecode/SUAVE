## @ingroup Attributes-Landscapes
# Landscape.py

# Created: Sep 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Attributes.Atmospheres import Earth

# ----------------------------------------------------------------------
#  Landscape Data Class
# ----------------------------------------------------------------------

## @ingroup Attributes-Landscapes
class Landscape(Data):
    """A basic Landscape.
    
    Assumptions:
    None
    
    Source:
    None
    """
    
    def __defaults__(self):
        """This sets the default values.
    
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """     
        
        self.tag        = 'Landscape'
        self.altitude   = 0.0        # m
        self.atmosphere = Earth.US_Standard_1976()
        self.delta_isa  = 0.0    
