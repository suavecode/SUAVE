# Aluminum.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
## @ingroup Attributes-Solids

from Solid import Solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Aluminum 6061-T6 Solid Class
#-------------------------------------------------------------------------------

class Aluminum(Solid):

    """ Physical Constants Specific to Aluminum 6061-T6"""

    def __defaults__(self):

        self.ultimateTensileStrength        = 310e6    *Units.Pa
        self.ultimateShearStrength          = 206e6    *Units.Pa
        self.ultimateBearingStrength        = 607e6    *Units.Pa
        self.yieldTensileStrength           = 276e6    *Units.Pa
        self.yieldShearStrength             = 206e6    *Units.Pa
        self.yieldBearingStrength           = 386e6    *Units.Pa
        self.minimumGageThickness           = 0.0      *Units.m
        self.density                        = 2700     *(Units.kg)/((Units.m)**3)
