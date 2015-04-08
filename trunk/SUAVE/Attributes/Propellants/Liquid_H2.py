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

        self.tag             = 'Liquid_H2'
        self.reactant        = 'O2'
        self.density         = 59.9                             # kg/m^3
        self.specific_energy = 141.86e6                         # J/kg
        self.energy_density  = 8491.0e6                         # J/m^3