# Honeycomb.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Carbon Fiber Honeycomb Core Solid Class
#-------------------------------------------------------------------------------

class Honeycomb(Solid):

    """ Physical Constants Specific to Carbon Fiber Honeycomb Core Material"""

    def __defaults__(self):

        self.ultimateTensileStrength        = 1e6       *Units.Pa
        self.ultimateShearStrength          = 1e6       *Units.Pa
        self.ultimateBearingStrength        = 1e6       *Units.Pa
        self.yieldTensileStrength           = 1e6       *Units.Pa
        self.yieldShearStrength             = 1e6       *Units.Pa
        self.yieldBearingStrength           = 1e6       *Units.Pa
        self.minimumGageThickness           = 6.5e-3    *Units.m
        self.density                        = 55        *(Units.kg)/((Units.m)**3)
