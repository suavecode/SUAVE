## @ingroup Attributes-Propellants
#Jet A1
#
# Created:  Unk 2013, SUAVE TEAM
# Modified: Apr 2015, SUAVE TEAM
#           Feb 2016, M.Vegh
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from .Propellant import Propellant

# ----------------------------------------------------------------------
#  Jet_A1 Propellant Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Propellants
class Jet_A1(Propellant):
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
        self.tag                       = 'Jet A1'
        self.reactant                  = 'O2'
        self.density                   = 804.0                            # kg/m^3 (15 C, 1 atm)
        self.specific_energy           = 43.15e6                          # J/kg
        self.energy_density            = 34692.6e6                        # J/m^3
        self.max_mass_fraction         = Data({'Air' : 0.0633, 'O2' : 0.3022})  # kg propellant / kg oxidizer
        self.temperatures.flash        = 311.15                           # K
        self.temperatures.autoignition = 483.15                           # K
        self.temperatures.freeze       = 226.15                           # K
        self.temperatures.boiling      = 0.0                              # K