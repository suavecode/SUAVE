## @ingroup Attributes-Gases
# CO2.py
# 
# Created:   Mar 2014, J. Sinsay
# Modified:  Jan 2016, M. Vegh      

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Gas import Gas

# ----------------------------------------------------------------------
#  CO2 Gas Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Gases
class CO2(Gas):
   """Holds constants for CO2.
    
   Assumptions:
   None
    
   Source:
   None
   """
   def __defaults__(self):
      """This sets the default values.

      Assumptions:
      None

      Source:
      Values commonly available

      Inputs:
      None

      Outputs:
      None

      Properties Used:
      None
      """            
      self.tag                   ='CO2'
      self.molecular_mass        = 44.01           # kg/kmol
      self.gas_specific_constant = 188.9                       # m^2/s^2-K, specific gas constant
      self.composition.CO2       = 1.0
 