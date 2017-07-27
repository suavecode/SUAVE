# Acrylic.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data

#-------------------------------------------------------------------------------
# Acrylic Solid Class
#-------------------------------------------------------------------------------

class Acrylic(Solid):

    """ Physical Constants Specific to Acrylic"""

    def __defaults__(self):

        self.UTS        = 75e6          # Ultimate Tensile Strength
        self.USS        = 55.2e6        # Ultimate Shear Strength
        self.UBS        = 0.0           # Ultimate Bearing Strength
        self.YTS        = 75e6          # Yield Tensile Strength
        self.YSS        = 55.2e6        # Yield Shear Strength
        self.YBS        = 0.0           # Yield Bearing Strength
        self.minThk     = 3.175e-3      # Miminum Gage Thickness
        self.density    = 1180          # Material Density
