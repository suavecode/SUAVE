""" Aviation_Gasoline.py: Physical properties of aviation gasoline """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Propellant import Propellant

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Aviation_Gasoline(Propellant):

    """ Physical properties of aviation gasoline """

    def __defaults__(self):

        self.tag='Aviation Gasoline'
        self.mass_density = 721.0               # kg/m^3
        self.specific_energy = 43.71e6          # J/kg
        
