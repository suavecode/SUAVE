## @ingroup Attributs-Solids

# Acrylic.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Acrylic Solid Class
#-------------------------------------------------------------------------------

class Acrylic(Solid):

    """ Physical Constants Specific to Polymethyl Methacrylate"""

    def __defaults__(self):

        self.ultimateTensileStrength        = 75e6          *Units.Pa
        self.ultimateShearStrength          = 55.2e6        *Units.Pa
        self.ultimateBearingStrength        = 0.0           *Units.Pa
        self.yieldTensileStrength           = 75e6          *Units.Pa
        self.yieldShearStrength             = 55.2e6        *Units.Pa
        self.yieldBearingStrength           = 0.0           *Units.Pa
        self.minimumGageThickness           = 3.175e-3      *Units.m
        self.density                        = 1180          *(Units.kg)/((Units.m)**3)
