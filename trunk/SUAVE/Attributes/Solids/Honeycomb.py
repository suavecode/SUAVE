# Honeycomb.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data

#-------------------------------------------------------------------------------
# Carbon Fiber Honeycomb Core Solid Class
#-------------------------------------------------------------------------------

class Honeycomb(Solid):

    """ Physical Constants Specific to Carbon Fiber Honeycomb Core Material"""

    def __defaults__(self):

        self.UTS        = 1e6       # Ultimate Tensile Strength
        self.USS        = 1e6       # Ultimate Shear Strength
        self.UBS        = 1e6       # Ultimate Bearing Strength
        self.YTS        = 1e6       # Yield Tensile Strength
        self.YSS        = 1e6       # Yield Shear Strength
        self.YBS        = 1e6       # Yield Bearing Strength
        self.minThk     = 6.5e-3    # Miminum Gage Thickness
        self.density    = 55       # Material Density
