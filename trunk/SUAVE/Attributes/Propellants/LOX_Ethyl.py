## @ingroup Attributes-Propellants
# LOX_Ethyl.py
# 
# Created:  Feb 2018, W. Maier
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Propellant import Propellant
# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Propellants
class LOX_Ethyl(Propellant):
    """Holds values for this propellant
    
    Assumptions:
    Stoichemetric O/F calculated CEA
    
    Source:
    NASA CEA code 
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
        self.tag                         = 'LOX_Eythl'
        self.molecular_weight            = 17.860
        self.isentropic_expansion_factor = 1.1613
        self.combustion_temperature      = 3252.31
        self.gas_specific_constant       = 8314.45986/self.molecular_weight