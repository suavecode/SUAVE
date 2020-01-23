## @ingroup Attributes-Cryogens
# Liquid N2
#
# Created:  Feb 2020, K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from .Cryogen import Cryogen

# ----------------------------------------------------------------------
#  Liquid N2 Cryogen Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Cryogens
class Liquid_N2(Cryogen):
    """Holds values for this cryogen
    
    Assumptions:
    None
    
    Source:
    None
    """

    def __defaults__(self):
        """This sets the default values.

        Assumptions:
        Ambient Pressure

        Source:

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """ 
        
        self.tag                        = 'Liquid_N2'
        self.density                    = 808.0             # [kg/m^3]
        self.temperatures.freeze        = 63.15             # K
        self.temperatures.boiling       = 77.355            # K