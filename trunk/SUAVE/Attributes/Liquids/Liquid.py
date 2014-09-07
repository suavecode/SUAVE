""" Liquid.py: Liquid data container class """
# Created by:     M. Vegh 12/12/13
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
# from SUAVE.Attributes.Constants import Constant
from SUAVE.Attributes.Constants import Composition

# ----------------------------------------------------------------------
#  Gas
# ----------------------------------------------------------------------

class Liquid(Data):

    """ Physical constants of a liquid """

    def __defaults__(self):

        self.molecular_mass  = 0.0               
        self.composition    = Data()
        self.composition.liquid = 1.0
        self.heat_of_vaporization=0.                 #heat of vaporization of water [J/kg]
        self.density=0.                   #density (kg/
        self.boiling_point=0.                   #boiling point [K]