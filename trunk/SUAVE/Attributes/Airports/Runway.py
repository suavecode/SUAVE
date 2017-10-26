## @ingroup Attributes-Airports
# Runway.py
# 
# Created:  Mar, 2014, SUAVE Team
# Modified: Jan, 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Runway Data Class
# ----------------------------------------------------------------------

## @ingroup Attributes-Airports  
class Runway(Data):
    """A basic runway.
    
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
        self.tag = 'Runway'
