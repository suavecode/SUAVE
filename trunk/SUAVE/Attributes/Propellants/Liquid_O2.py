""" Liquid_O2.py: Physical properties of liquid O2 """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Propellant import Propellant

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Liquid_O2(Propellant):

    """ Physical properties of liquid O2 """

    def __defaults__(self): ### placeholder - data not accurate 

        self.tag = 'O2 Liquid'
        self.reactant = 'H2'
        self.density = 0.0                                    # kg/m^3
        self.specific_energy = 0.0                         # J/kg
        self.energy_density = 0.0                         # J/m^3
        self.max_mass_fraction = {'Air' : 0.0, 'O2' : 0.0}  # kg propellant / kg oxidizer
        self.temperatures = Data()
        self.temperatures.flash = 0.0                           # K
        self.temperatures.autoignition = 0.0                    # K
        self.temperatures.freeze = 0.0                          # K
        self.temperatures.boiling = 0.0                         # K