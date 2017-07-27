# Rib.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data

#-------------------------------------------------------------------------------
# Solid Data Class
#-------------------------------------------------------------------------------

class Solid(Data):

    """ Physical Constants of a Solid"""

    def __defaults__(self):

        self.UTS        = 310e6    # Ultimate Tensile Strength
        self.USS        = 206e6    # Ultimate Shear Strength
        self.UBS        = 607e6    # Ultimate Bearing Strength
        self.YTS        = 276e6    # Yield Tensile Strength
        self.YSS        = 206e6    # Yield Shear Strength
        self.YBS        = 386e6    # Yield Bearing Strength
        self.minThk     = 1.5e-3   # Miminum Gage Thickness
        self.minWidth   = 25.4e-3  # Minimum Gage Widths
        self.density    = 2700     # Material Density
