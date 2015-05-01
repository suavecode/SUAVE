# Propellant.py
# 
# Created:  Unk 2013, SUAVE TEAM
# Modified: Apr 2015, SUAVE TEAM


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Propellant(Data):

    """ Physical properties of propellants """

    def __defaults__(self):

        self.tag                       = 'Propellant'
        self.reactant                  = 'O2'
        self.density                   = 0.0                       # kg/m^3
        self.specific_energy           = 0.0                       # MJ/kg
        self.energy_density            = 0.0                       # MJ/m^3
        self.max_mass_fraction         = {'Air' : 0.0, 'O2' : 0.0} # kg propellant / kg oxidizer
        self.temperatures              = Data()
        self.temperatures.flash        = 0.0                       # K
        self.temperatures.autoignition = 0.0                       # K
        self.temperatures.freeze       = 0.0                       # K
        self.temperatures.boiling      = 0.0                       # K