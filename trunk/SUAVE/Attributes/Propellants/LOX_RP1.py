## @ingroup Attributes-Propellants
# LOX_RP1.py
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
class LOX_RP1(Propellant):
    """Holds values for this propellant
    
    Assumptions:
    At an O/F ratio 2.27
    
    
    Source:
    Sutton, Rocket Propulsion Elements
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
        self.molecular_weight            = 22.193 # [kg/kmol]
        self.isentropic_expansion_factor = 1.1505
        self.combustion_temperature      = 3545.69*Units.kelvin             #[k]
        self.gas_specific_constant       = 8314.45986/self.molecular_weight*Units['J/(kg*K)']  # [J/(Kg-K)]