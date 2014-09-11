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

        self.molecular_mass  = 0.0    
        self.gas_specific_constant              = 0.0              
        self.composition = Data()
        self.composition.gas = 1.0
