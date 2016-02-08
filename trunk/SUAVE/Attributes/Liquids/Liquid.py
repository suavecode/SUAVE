# Liquid.py

# Created:  Dec, 2013, M. Vegh
# Modified: Jan, 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Liquid Data Class
# ----------------------------------------------------------------------

class Liquid(Data):

    """ Physical constants of a liquid """

    def __defaults__(self):

        self.molecular_mass  = 0.0               
        self.composition    = Data()
        self.composition.liquid = 1.0
        self.heat_of_vaporization=0.            #heat of vaporization of water [J/kg]
        self.density=0.                         #density (kg/
        self.boiling_point=0.                   #boiling point [K]