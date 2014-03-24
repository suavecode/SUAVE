""" Gas.py: Gas data container class """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
# from SUAVE.Attributes.Constants import Constant
from SUAVE.Attributes.Constants import Composition

# ----------------------------------------------------------------------
#  Gas
# ----------------------------------------------------------------------

class Gas(Data):

    """ Physical constants of a gas """

    def __defaults__(self):

        self.MolecularMass  = 0.0    
        self.R              = 0.0              
        self.Composition    = Composition( Gas = 1.0 )
