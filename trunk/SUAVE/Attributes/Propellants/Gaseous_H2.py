#Gaseous_H2.py: Physical properties of gaseous H2 for propulsion use
#
# Created:  Unk 2013, SUAVE TEAM
# Modified: Apr 2015, SUAVE TEAM
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data
from Propellant import Propellant
from SUAVE.Attributes.Constants import Composition
# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Gaseous_H2(Propellant):

    """ Physical properties of gaseous H2 for propulsion use; reactant = O2 """

    def __defaults__(self):

        self.tag                       = 'H2 Gas'
        self.reactant                  = 'O2'
        self.specific_energy           = 141.86e6                           # J/kg
        self.energy_density            = 5591.13e6                          # J/m^3
        self.max_mass_fraction         = {'Air' : 0.013197, 'O2' : 0.0630}  # kg propellant / kg oxidizer
        self.temperatures              = Data()
        self.temperatures.flash        = 0.0                               # K
        self.temperatures.autoignition = 0.0                               # K
        self.temperatures.freeze       = 0.0                               # K
        self.temperatures.boiling      = 0.0                               # K

        # gas properties
        self.composition               = Composition( H2 = 1.0 )
        self.molecular_mass            = 2.016                             # kg/kmol
        self.gas_constant              = 4124.0                            # J/kg-K              
        self.pressure                  = 700e5                             # Pa
        self.temperature               = 293.0                             # K
        self.compressibility_factor    = 1.4699                            # compressibility factor
        self.density                   = 39.4116                           # kg/m^3
