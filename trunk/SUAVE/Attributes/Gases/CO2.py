# Constants.py: Physical constants and helepr functions
# 
# Created By:       J. Sinsay
# Updated:          M. Colonno   04/09/2013
#                   T. Lukaczyk  06/23/2013

""" SUAVE Data Class for Constants """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# classes
from Gas import Gas
from SUAVE.Attributes.Constants import Composition
from SUAVE.Core import Data, Data_Exception, Data_Warning

# ----------------------------------------------------------------------
#  CO2
# ----------------------------------------------------------------------

class CO2(Gas):
    """ Physcial constants specific to CO2 """
    def __defaults__(self):
        self.molecular_mass = 44.01           # kg/kmol
        self.gas_specific_constant = 188.9                       # m^2/s^2-K, specific gas constant
        self.composition = Data()
        self.composition.CO2 = 1.0
 