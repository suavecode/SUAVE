# Liquid_Natural_Gas.py
# 
# Created:  Unk 2013, SUAVE TEAM
# Modified: Apr 2015, SUAVE TEAM
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Propellant import Propellant

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Liquid_Natural_Gas(Propellant):

    """ Physical properties of LNG; reactant = O2 """

    def __defaults__(self):

        self.tag             = 'Liquid_Natural_Gas'
        self.reactant        = 'O2'
        self.density         = 414.2                            # kg/m^3 
        self.specific_energy = 53.6e6                           # J/kg
        self.energy_density  = 22200.0e6                        # J/m^3