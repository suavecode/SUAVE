# Nickel.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data

#-------------------------------------------------------------------------------
# Cold Rolled Nickel/Cobalt Chromoly Alloy Solid Class
#-------------------------------------------------------------------------------

class Nickel(Solid):

    """ Physical Constants Specific to Cold Rolled Nickel/Cobalt Chromoly Alloy"""

    def __defaults__(self):

        self.UTS        = 1830e6   # Ultimate Tensile Strength
        self.USS        = 1050e6     # Ultimate Shear Strength
        self.UBS        = 1830e6   # Ultimate Bearing Strength
        self.YTS        = 1550e6   # Yield Tensile Strength
        self.YSS        = 1050e6     # Yield Shear Strength
        self.YBS        = 1550e6   # Yield Bearing Strength
        self.minThk     = 0        # Miminum Gage Thickness
        self.density    = 8430     # Material Density
