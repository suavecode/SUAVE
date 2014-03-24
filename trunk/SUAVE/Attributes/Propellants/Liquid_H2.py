""" Liquid_H2.py: Physical properties of liquid H2 for propulsion use """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Propellant import Propellant

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Liquid_H2(Propellant):

    """ Physical properties of liquid H2 for propulsion use; reactant = O2 """

    def __defaults__(self):

        self.tag = 'H2 Liquid'
        self.reactant = 'O2'
        self.density = 70.99                                    # kg/m^3
        self.specific_energy = 141.86e6                         # J/kg
        self.energy_density = 10071.0e6                         # J/m^3
        self.max_mass_fraction = {'Air' : 0.013197, 'O2' : 0.0630}  # kg propellant / kg oxidizer
        self.temperatures = Data()
        self.temperatures.flash = 0.0                           # K
        self.temperatures.autoignition = 0.0                    # K
        self.temperatures.freeze = 0.0                          # K
        self.temperatures.boiling = 0.0                         # K