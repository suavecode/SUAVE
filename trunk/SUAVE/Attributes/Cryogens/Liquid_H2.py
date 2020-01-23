## @ingroup Attributes-Propellants
# Liquid H2
#
# Created:  Feb 2020, K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from .Cryogen import Cryogen

# ----------------------------------------------------------------------
#  Liquid H2 Cryogen Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Cryogens
class Liquid_H2(Cryogen):
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
        
        self.tag                        = 'Liquid_H2'
        self.density                    = 59.9              # [kg/m^3] 
        self.specific_energy            = 141.86e6          # [J/kg] 
        self.energy_density             = 8491.0e6          # [J/m^3]
        self.temperatures.freeze        = 13.99             # K
        self.temperatures.boiling       = 20.271            # K