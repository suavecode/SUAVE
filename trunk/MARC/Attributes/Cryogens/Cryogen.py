## @ingroup Attributes-Cryogens
# Cryogen.py
# 
# Created:  Feb 2020,  K. Hamilton - Through New Zealand Ministry of Business Innovation and Employment Research Contract RTVU2004


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from MARC.Core import Data

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Cryogens
class Cryogen(Data):
    """Holds values for a cryogen
    
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
        Values commonly available

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """    
        self.tag                       = 'Cryogen'
        self.density                   = 0.0                       # kg/m^3
        self.specific_energy           = 0.0                       # MJ/kg
        self.energy_density            = 0.0                       # MJ/m^3
        self.temperatures              = Data()
        self.temperatures.freeze       = 0.0                       # K
        self.temperatures.boiling      = 0.0                       # K