# Steel.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data

#-------------------------------------------------------------------------------
# AISI 4340 Steel Solid Class
#-------------------------------------------------------------------------------

class Steel(Solid):

    """ Physical Constants Specific to AISI 4340 Steel"""

    def __defaults__(self):

        self.UTS        = 1110e6    # Ultimate Tensile Strength
        self.USS        = 825e6     # Ultimate Shear Strength
        self.UBS        = 1110e6    # Ultimate Bearing Strength
        self.YTS        = 710e6     # Yield Tensile Strength
        self.YSS        = 410e6     # Yield Shear Strength
        self.YBS        = 710e6     # Yield Bearing Strength
        self.minThk     = 0.0       # Miminum Gage Thickness
        self.density    = 7850      # Material Density
