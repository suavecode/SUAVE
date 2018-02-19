# BiCF.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Bi-Directional Carbon Fiber Solid Class
#-------------------------------------------------------------------------------

class BiCF(Solid):

    """ Physical Constants of a Specific to Bi-Directional Carbon Fiber"""

    def __defaults__(self):

        self.ultimateTensileStrength        = 600e6    *Units.Pa
        self.ultimateShearStrength          = 90e6     *Units.Pa
        self.ultimateBearingStrength        = 600e6    *Units.Pa
        self.yieldTensileStrength           = 600e6    *Units.Pa
        self.yieldShearStrength             = 90e6     *Units.Pa
        self.yieldBearingStrength           = 600e6    *Units.Pa
        self.minimumGageThickness           = 420e-6   *Units.m
        self.density                        = 1600     *(Units.kg)/((Units.m)**3)
