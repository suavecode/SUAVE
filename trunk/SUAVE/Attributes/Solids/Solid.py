# Solid.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Data

#-------------------------------------------------------------------------------
# Solid Data Class
#-------------------------------------------------------------------------------

class Solid(Data):

    """ Physical Constants of a Solid"""

    def __defaults__(self):

        self.UTS        = 0.0   # Ultimate Tensile Strength
        self.USS        = 0.0   # Ultimate Shear Strength
        self.UBS        = 0.0   # Ultimate Bearing Strength
        self.YTS        = 0.0   # Yield Tensile Strength
        self.YSS        = 0.0   # Yield Shear Strength
        self.YBS        = 0.0   # Yield Bearing Strength
        self.minThk     = 0.0   # Miminum Gage Thickness
        self.density    = 0.0   # Material Density 
