# CO2.py
# 
# Created:   Mar, 2014, J. Sinsay
# Modified:  Jan, 2016, M. Vegh      

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from Gas import Gas

# ----------------------------------------------------------------------
#  CO2 Gas Class
# ----------------------------------------------------------------------

class CO2(Gas):
    """ Physical constants specific to CO2 """
    def __defaults__(self):
        self.molecular_mass = 44.01           # kg/kmol
        self.gas_specific_constant = 188.9                       # m^2/s^2-K, specific gas constant
        self.composition.CO2 = 1.0
 