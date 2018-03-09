## @ingroup Attributes-Solids

# Nickel.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Cold Rolled Nickel/Cobalt Chromoly Alloy Solid Class
#-------------------------------------------------------------------------------

class nickel(solid):

    """ Physical Constants Specific to Cold Rolled Nickel/Cobalt Chromoly Alloy"""

    def __defaults__(self):

        self.ultimate_tensile_strength        = 1830e6   *Units.Pa
        self.ultimate_shear_strength          = 1050e6   *Units.Pa
        self.ultimate_bearing_strength        = 1830e6   *Units.Pa
        self.yield_tensile_strength           = 1550e6   *Units.Pa
        self.yield_shear_strength             = 1050e6   *Units.Pa
        self.yield_bearing_strength           = 1550e6   *Units.Pa
        self.minimum_gage_thickness           = 0        *Units.m
        self.density                          = 8430     *(Units.kg)/((Units.m)**3)
