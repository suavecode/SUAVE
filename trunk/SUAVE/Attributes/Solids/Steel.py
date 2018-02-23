## @ingroup Attributes-Solids

# Steel.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# AISI 4340 Steel Solid Class
#-------------------------------------------------------------------------------

class Steel(Solid):

    """ Physical Constants Specific to AISI 4340 Steel"""

    def __defaults__(self):

        self.ultimateTensileStrength        = 1110e6    *Units.Pa
        self.ultimateShearStrength          = 825e6     *Units.Pa
        self.ultimateBearingStrength        = 1110e6    *Units.Pa
        self.yieldTensileStrength           = 710e6     *Units.Pa
        self.yieldShearStrength             = 410e6     *Units.Pa
        self.yieldBearingStrength           = 710e6     *Units.Pa
        self.minimumGageThickness           = 0.0       *Units.m
        self.density                        = 7850      *(Units.kg)/((Units.m)**3)
