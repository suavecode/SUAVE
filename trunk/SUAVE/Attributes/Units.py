""" Implements base unit conversion style programming
    by monkey patching Pint
"""

# Created: T. Lukaczyk, Feb 2014
# Modified:


# ------------------------------------------------------------
#   Imports
# ------------------------------------------------------------

from SUAVE.Plugins.pint import UnitRegistry

Units = UnitRegistry()


# ------------------------------------------------------------
#   Monkey Patching
# ------------------------------------------------------------
 
# TODO - non-linear or offset conversions (ie Temp) 
 
# multiplication covnverts in to base unit
def __rmul__(self,other):
    self.ito_base_units()
    return other * self.magnitude

# division converts out of base unit
def __rdiv__(self,other):
    self.ito_base_units()
    return other / self.magnitude

# yay monkey patching!
Units.Quantity.__mul__      = __rmul__
Units.Quantity.__rmul__     = __rmul__
Units.Quantity.__div__      = __rdiv__
Units.Quantity.__truediv__  = __rdiv__
Units.Quantity.__rdiv__     = __rdiv__
Units.Quantity.__rtruediv__ = __rdiv__
Units.Quantity.__getattr__  = getattr
Units.Quantity.__array_prepare__ = None
Units.Quantity.__array_wrap__    = None

# doc string
Units.__doc__ = \
""" SUAVE.Attributes.Units()
    Unit conversion toolbox
    Works by converting values in to and out of the base unit
    
    Important Note and Warning - 
        This does not enforce unit consistency!!!
        Unit consistency is the responsibility of the user
    
    Usage:
      from SUAVE.Attributes import Units
      a = 4. * Units.mm  # convert in to base unit
      b = a  / Units.mm  # convert out of base unit
      
    Comments:
      Retreving an attribute of Units (ie Units.mm) returns 
      the conversion ratio to the base unit.  So in the above
      example Units.mm = 0.001, which is the conversion ratio
      to meters.  Thus the * (multiplication) operation converts 
      from the current units to the base units and / (division) 
      operation converts from the base units to the desired units.
     
    Base Units:
      mass        : kilogram
      length      : meters
      time        : seconds
      temperature : Kelvin
      angle       : radian
      current     : Ampere
      luminsoity  : candela
      
    
    Based on the Pint package, included in SUAVE.Plugins
    https://pint.readthedocs.org/en/latest/
    
"""


# ------------------------------------------------------------
#   Unit Tests
# ------------------------------------------------------------

if __name__ == '__main__':
    
    a = 4. * Units.kilogram
    b = a / Units.gram
    
    print a
    print b
    
    import numpy as np
    
    c = np.array([3.,4.,6.])
    c = c * Units.kg
    d = c / Units.g
    
    print c
    print d
    