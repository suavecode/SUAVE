# Paint.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Paint and/or Vinyl Surface Convering Solid Class
#-------------------------------------------------------------------------------

class Paint(Solid):

    """ Physical Constants Specific to Paint and/or Vinyl Surface Covering"""

    def __defaults__(self):

        self.ultimateTensileStrength        = 0.0       *Units.Pa
        self.ultimateShearStrength          = 0.0       *Units.Pa
        self.ultimateBearingStrength        = 0.0       *Units.Pa
        self.yieldTensileStrength           = 0.0       *Units.Pa
        self.yieldShearStrength             = 0.0       *Units.Pa
        self.yieldBearingStrength           = 0.0       *Units.Pa
        self.minimumGageThickness           = 150e-6    *Units.m
        self.density                        = 1800       *(Units.kg)/((Units.m)**3)
