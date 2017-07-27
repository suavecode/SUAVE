# UniCF.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data

#-------------------------------------------------------------------------------
# Uni-Directional Carbon Fiber Solid Class
#-------------------------------------------------------------------------------

class UniCF(Solid):

    """ Physical Constants Specific to Uni-Directional Carbon Fiber"""

    def __defaults__(self):

        self.UTS        = 1500e6   # Ultimate Tensile Strength
        self.USS        = 70e6     # Ultimate Shear Strength
        self.UBS        = 1500e6   # Ultimate Bearing Strength
        self.YTS        = 1500e6   # Yield Tensile Strength
        self.YSS        = 70e6     # Yield Shear Strength
        self.YBS        = 1500e6   # Yield Bearing Strength
        self.minThk     = 420e-6   # Miminum Gage Thickness
        self.density    = 1600     # Material Density
