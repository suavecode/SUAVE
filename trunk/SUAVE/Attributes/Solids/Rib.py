# Rib.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Solid Data Class
#-------------------------------------------------------------------------------

class Rib(Solid):

    """ Physical Constants of a Solid"""

    def __defaults__(self):

        self.ultimateTensileStrength        = 310e6    *Units.Pa
        self.ultimateShearStrength          = 206e6    *Units.Pa
        self.ultimateBearingStrength        = 607e6    *Units.Pa
        self.yieldTensileStrength           = 276e6    *Units.Pa
        self.yieldShearStrength             = 206e6    *Units.Pa
        self.yieldBearingStrength           = 386e6    *Units.Pa
        self.minimumGageThickness           = 1.5e-3   *Units.m
        self.minWidth                       = 25.4e-3  # Minimum Gage Widths
        self.density                        = 2700     *(Units.kg)/((Units.m)**3)
