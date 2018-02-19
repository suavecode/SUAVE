# UniCF.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Uni-Directional Carbon Fiber Solid Class
#-------------------------------------------------------------------------------

class UniCF(Solid):

    """ Physical Constants Specific to Uni-Directional Carbon Fiber"""

    def __defaults__(self):

        self.ultimateTensileStrength        = 1500e6   *Units.Pa
        self.ultimateShearStrength          = 70e6     *Units.Pa
        self.ultimateBearingStrength        = 1500e6   *Units.Pa
        self.yieldTensileStrength           = 1500e6   *Units.Pa
        self.yieldShearStrength             = 70e6     *Units.Pa
        self.yieldBearingStrength           = 1500e6   *Units.Pa
        self.minimumGageThickness           = 420e-6   *Units.m
        self.density                        = 1600     *(Units.kg)/((Units.m)**3)
