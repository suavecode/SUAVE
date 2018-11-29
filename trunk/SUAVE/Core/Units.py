## @ingroup Core
# Units.py
#
# Created:  Feb 2014, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald

""" Implements base unit conversion style programming
    by monkey patching Pint
"""


# ------------------------------------------------------------
#   Imports
# ------------------------------------------------------------

from SUAVE.Plugins.pint import UnitRegistry
from SUAVE.Plugins.pint.quantity import _Quantity

Units = UnitRegistry()


# ------------------------------------------------------------
#   Monkey Patching
# ------------------------------------------------------------
 
# multiplication covnverts in to base unit
## @ingroup Core
def __rmul__(self,other):
    """ Override the basic python multiplication for Units

        Assumptions:
        N/A

        Source:
        N/A

        Inputs:
        Other

        Outputs:
        Converted into Base Units!

        Properties Used:
        N/A    
    """      
    if isinstance(other,_Quantity):
        return _Quantity.__rmul__(self,other)
    else:
        self._magnitude = other
        self.ito_base_units()
        return self.magnitude

# division converts out of base unit
## @ingroup Core
def __rdiv__(self,other):
    """ Override the basic python division for Units

        Assumptions:
        N/A

        Source:
        N/A

        Inputs:
        Other

        Outputs:
        Converted from Base Units!

        Properties Used:
        N/A    
    """       
    if isinstance(other,_Quantity):
        return _Quantity.__truediv__(self,other)
    else:
        units = str(self._units)
        self.ito_base_units()
        self._magnitude = other
        self.ito(units)
        return self.magnitude

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
      from SUAVE.Core import Units
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
    
    import numpy as np
    
    x = Units['miles/hour']
    y = Units.miles / Units.hour
    print(x)
    print(y)
    
    x = Units['slug/ft**3']
    y = Units.slug / Units.ft**3
    print(x)
    print(y)    
    
    a = 4. * Units.kilogram
    b = 5. * Units.gram
    v = np.array([3.,4.,6.]) * Units['miles/hour']
    t = 100 * Units.degF
    
    print(a)
    print(b)
    print(v)
    print(t)
    
    a = a / Units.g
    b = b / Units.g
    v = v / Units['miles/hour']
    t = t / Units.degF
    
    print(a)
    print(b)
    print(v)
    print(t)