## @ingroup Attributes-Propellants
# Aviation_Gasoline.py
# 
# Created:  Unk 2013, SUAVE TEAM
# Modified: Apr 2015, SUAVE TEAM

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Propellant import Propellant

# ----------------------------------------------------------------------
#  Aviation_Gasoline Propellant Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Propellants
class Aviation_Gasoline(Propellant):
    """Contains density and specific energy values for this propellant
    
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
        self.tag='Aviation Gasoline'
        self.density         = 721.0            # kg/m^3
        self.specific_energy = 43.71e6          # J/kg
        
