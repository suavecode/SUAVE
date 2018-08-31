## @ingroup Attributes-Propellants
# LOX_LH2.py
# 
# Created:  Feb 2018, W. Maier
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from .Propellant import Propellant
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Propellants
class LOX_LH2(Propellant):
    """Holds values for this propellant
    
    Assumptions:
    At an O/F ratio 5.50
    
    
    Source:
    Sutton, Rocket Propulsion Elements
    Using CEA
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
        self.tag                         = 'LOX_RP1'
        self.molecular_weight            = 12.644                             # [kg/kmol]
        self.isentropic_expansion_factor = 1.145
        self.combustion_temperature      = 3331.0*Units.kelvin                # [K]                      
        self.gas_specific_constant       = (8314.45986/self.molecular_weight)*Units['J/(kg*K)'] # [J/(kg-K)]