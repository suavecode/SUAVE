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

    """ Physical constants of a gas """

    def __defaults__(self):

        self.MolecularMass  = 0.0               
        self.Composition    = Composition( Liquid = 1.0 )
        self.h_vap=0.                 #heat of vaporization of water [J/kg]
        self.rho=0.                   #density (kg/
        self.T_vap=0.                   #boiling point [K]