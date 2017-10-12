## @ingroup Attributes-Propellants
# Liquid_H2.py:
#
# Created:  Unk 2013, SUAVE TEAM
# Modified: Apr 2015, SUAVE TEAM
#           Feb 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Propellant import Propellant

# ----------------------------------------------------------------------
#  Liquid_H2 Propellant Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Propellants
class Liquid_H2(Propellant):
    """Holds values for this propellant
    
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
        self.tag             = 'Liquid_H2'
        self.reactant        = 'O2'
        self.density         = 59.9                             # kg/m^3
        self.specific_energy = 141.86e6                         # J/kg
        self.energy_density  = 8491.0e6                         # J/m^3