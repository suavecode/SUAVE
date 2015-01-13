""" Propellant.py: Physical properties of propellants """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# classes
# from SUAVE.Attributes.Gases import Gas
from SUAVE.Core import Data
# from SUAVE.Attributes.Constants import Composition

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Propellant(Data):

    """ Physical properties of propellants """

    def __defaults__(self):

        self.tag = 'Propellant'
        self.reactant = 'O2'
        self.density = 0.0                                      # kg/m^3
        self.specific_energy = 0.0                              # MJ/kg
        self.energy_density = 0.0                               # MJ/m^3
        self.max_mass_fraction = {'Air' : 0.0, 'O2' : 0.0}      # kg propellant / kg oxidizer
        self.temperatures = Data()
        self.temperatures.flash = 0.0                           # K
        self.temperatures.autoignition = 0.0                    # K
        self.temperatures.freeze = 0.0                          # K
        self.temperatures.boiling = 0.0                         # K