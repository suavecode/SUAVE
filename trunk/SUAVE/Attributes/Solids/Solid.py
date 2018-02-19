# Solid.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Solid Data Class
#-------------------------------------------------------------------------------

class Solid(Data):

    """ Physical Constants of a Solid"""

    def __defaults__(self):

        self.ultimateTensileStrength        = 0.0   *Units.Pa
        self.ultimateShearStrength          = 0.0   *Units.Pa
        self.ultimateBearingStrength        = 0.0   *Units.Pa
        self.yieldTensileStrength           = 0.0   *Units.Pa
        self.yieldShearStrength             = 0.0   *Units.Pa
        self.yieldBearingStrength           = 0.0   *Units.Pa
        self.minimumGageThickness           = 0.0   *Units.m
        self.density                        = 0.0   *(Units.kg)/((Units.m)**3) 
