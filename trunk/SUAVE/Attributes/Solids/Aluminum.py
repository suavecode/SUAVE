# Aluminum.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data

#-------------------------------------------------------------------------------
# Aluminum 6061-T6 Solid Class
#-------------------------------------------------------------------------------

class Aluminum(Solid):

    """ Physical Constants Specific to Aluminum 6061-T6"""

    def __defaults__(self):

        self.UTS        = 310e6    # Ultimate Tensile Strength
        self.USS        = 206e6    # Ultimate Shear Strength
        self.UBS        = 607e6    # Ultimate Bearing Strength
        self.YTS        = 276e6    # Yield Tensile Strength
        self.YSS        = 206e6    # Yield Shear Strength
        self.YBS        = 386e6    # Yield Bearing Strength
        self.minThk     = 0.0      # Miminum Gage Thickness
        self.density    = 2700     # Material Density
