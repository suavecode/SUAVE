## @ingroup Attributes-Solids

# Nickel.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Cold Rolled Nickel/Cobalt Chromoly Alloy Solid Class
#-------------------------------------------------------------------------------

class Nickel(Solid):

    """ Physical Constants Specific to Cold Rolled Nickel/Cobalt Chromoly Alloy"""

    def __defaults__(self):

        self.ultimateTensileStrength        = 1830e6   *Units.Pa
        self.ultimateShearStrength          = 1050e6   *Units.Pa
        self.ultimateBearingStrength        = 1830e6   *Units.Pa
        self.yieldTensileStrength           = 1550e6   *Units.Pa
        self.yieldShearStrength             = 1050e6   *Units.Pa
        self.yieldBearingStrength           = 1550e6   *Units.Pa
        self.minimumGageThickness           = 0        *Units.m
        self.density                        = 8430     *(Units.kg)/((Units.m)**3)
