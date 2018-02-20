# Epoxy.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Hardened Epoxy Resin Solid Class
#-------------------------------------------------------------------------------

class Epoxy(Solid):

    """ Physical Constants Specific to Hardened Expoy Resin"""

    def __defaults__(self):

        self.ultimateTensileStrength        = 0.0      *Units.Pa
        self.ultimateShearStrength          = 0.0      *Units.Pa
        self.ultimateBearingStrength        = 0.0      *Units.Pa
        self.yieldTensileStrength           = 0.0      *Units.Pa
        self.yieldShearStrength             = 0.0      *Units.Pa
        self.yieldBearingStrength           = 0.0      *Units.Pa
        self.minimumGageThickness           = 250e-6   *Units.m
        self.density                        = 1800     *(Units.kg)/((Units.m)**3)
