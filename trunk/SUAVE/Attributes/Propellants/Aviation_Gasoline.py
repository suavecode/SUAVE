# Aviation_Gasoline.py
# 
# Created:  Unk 2013, SUAVE TEAM
# Modified: Apr 2015, SUAVE TEAM

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Propellant import Propellant

# ----------------------------------------------------------------------
#  Aviation_Gasoline Propellant Class
# ----------------------------------------------------------------------

class Aviation_Gasoline(Propellant):

    """ Physical properties of aviation gasoline """

    def __defaults__(self):

        self.tag='Aviation Gasoline'
        self.mass_density    = 721.0            # kg/m^3
        self.specific_energy = 43.71e6          # J/kg
        
