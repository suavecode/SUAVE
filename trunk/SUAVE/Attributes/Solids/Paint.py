# Paint.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data

#-------------------------------------------------------------------------------
# Paint and/or Vinyl Surface Convering Solid Class
#-------------------------------------------------------------------------------

class Paint(Solid):

    """ Physical Constants Specific to Paint and/or Vinyl Surface Covering"""

    def __defaults__(self):

        self.UTS        = 0.0       # Ultimate Tensile Strength
        self.USS        = 0.0       # Ultimate Shear Strength
        self.UBS        = 0.0       # Ultimate Bearing Strength
        self.YTS        = 0.0       # Yield Tensile Strength
        self.YSS        = 0.0       # Yield Shear Strength
        self.YBS        = 0.0       # Yield Bearing Strength
        self.minThk     = 150e-6    # Miminum Gage Thickness
        self.density    = 1800       # Material Density
