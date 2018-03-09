## @ingroup Attributes-Solids

# Honeycomb.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Carbon Fiber Honeycomb Core Solid Class
#-------------------------------------------------------------------------------

class honeycomb(solid):

    """ Physical Constants Specific to Carbon Fiber Honeycomb Core Material"""

    def __defaults__(self):

        self.ultimate_tensile_strength        = 1e6       *Units.Pa
        self.ultimate_shear_strength          = 1e6       *Units.Pa
        self.ultimate_bearing_strength        = 1e6       *Units.Pa
        self.yield_tensile_strength           = 1e6       *Units.Pa
        self.yield_shear_strength             = 1e6       *Units.Pa
        self.yield_bearing_strength           = 1e6       *Units.Pa
        self.minimum_gage_thickness           = 6.5e-3    *Units.m
        self.density                          = 55        *(Units.kg)/((Units.m)**3)
