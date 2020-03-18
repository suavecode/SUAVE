#Atmosphere.py

# Created:  Mar 2014, SUAVE Team
# Modified: Jan 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

## @ingroup Attributes-Atmospheres
from SUAVE.Attributes.Constants import Constant #, Composition
from SUAVE.Core import Data


# ----------------------------------------------------------------------
#  Atmosphere Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Atmospheres
class Atmosphere(Constant):
    """The base atmosphere class.
    
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
        self.tag = 'Constant-property atmosphere'
        self.composition           = Data()
        self.composition.gas       = 1.0
