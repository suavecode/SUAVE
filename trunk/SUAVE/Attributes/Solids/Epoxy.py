# Epoxy.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid Import Solid
from SUAVE.Core import Data

#-------------------------------------------------------------------------------
# Hardened Epoxy Resin Solid Class
#-------------------------------------------------------------------------------

class Epoxy(Solid):

    """ Physical Constants Specific to Hardened Expoy Resin"""

    def __defaults__(self):

        self.UTS        = 0.0      # Ultimate Tensile Strength
        self.USS        = 0.0      # Ultimate Shear Strength
        self.UBS        = 0.0      # Ultimate Bearing Strength
        self.YTS        = 0.0      # Yield Tensile Strength
        self.YSS        = 0.0      # Yield Shear Strength
        self.YBS        = 0.0      # Yield Bearing Strength
        self.minThk     = 250e-6   # Miminum Gage Thickness
        self.density    = 1800     # Material Density
