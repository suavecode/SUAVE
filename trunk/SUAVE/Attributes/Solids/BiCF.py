# BiCF.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data

#-------------------------------------------------------------------------------
# Bi-Directional Carbon Fiber Solid Class
#-------------------------------------------------------------------------------

class BiCF(Solid):

    """ Physical Constants of a Specific to Bi-Directional Carbon Fiber"""

    def __defaults__(self):

        self.UTS        = 600e6    # Ultimate Tensile Strength
        self.USS        = 90e6     # Ultimate Shear Strength
        self.UBS        = 600e6    # Ultimate Bearing Strength
        self.YTS        = 600e6    # Yield Tensile Strength
        self.YSS        = 90e6     # Yield Shear Strength
        self.YBS        = 600e6    # Yield Bearing Strength
        self.minThk     = 420e-6   # Miminum Gage Thickness
        self.density    = 1600     # Material Density
