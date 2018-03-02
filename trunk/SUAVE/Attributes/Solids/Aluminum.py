# Aluminum.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
## @ingroup Attributes-Solids

from Solid import Solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Aluminum 6061-T6 Solid Class
#-------------------------------------------------------------------------------

class Aluminum(Solid):

    """ Physical Constants Specific to Aluminum 6061-T6"""

    def __defaults__(self):

        self.ultimate_tensile_strength        = 310e6    *Units.Pa
        self.ultimate_shear_strength          = 206e6    *Units.Pa
        self.ultimate_bearing_strength        = 607e6    *Units.Pa
        self.yield_tensile_strength           = 276e6    *Units.Pa
        self.yield_shear_strength             = 206e6    *Units.Pa
        self.yield_bearing_strength           = 386e6    *Units.Pa
        self.minimum_gage_thickness           = 0.0      *Units.m
        self.density                          = 2700     *(Units.kg)/((Units.m)**3)
