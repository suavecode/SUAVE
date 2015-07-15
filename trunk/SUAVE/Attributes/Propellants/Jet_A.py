#Jet A
#
# Created:  Unk 2013, SUAVE TEAM
# Modified: Apr 2015, SUAVE TEAM

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Propellant import Propellant
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Jet_A(Propellant):

    """ Physical properties of Jet A; reactant = O2 """

    def __defaults__(self):

        self.tag                       = 'Jet_A'
        self.reactant                  = 'O2'
        self.density                   = 820.0                          # kg/m^3 (15 C, 1 atm)
        self.specific_energy           = 43.02e6                        # J/kg
        self.energy_density            = 35276.4e6                      # J/m^3
        self.max_mass_fraction         = {'Air' : 0.0633, \
                                          'O2' : 0.3022}                # kg propellant / kg oxidizer

        # ciritical temperatures
        self.temperatures              = Data()
        self.temperatures.flash        = 311.15                 # K
        self.temperatures.autoignition = 483.15                 # K
        self.temperatures.freeze       = 233.15                 # K
        self.temperatures.boiling      = 0.0                    # K