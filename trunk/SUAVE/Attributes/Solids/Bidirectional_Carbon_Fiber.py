## @ingroup Attributes-Solids

# Bidirectional_Carbon_Fiber.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Bi-Directional Carbon Fiber Solid Class
#-------------------------------------------------------------------------------

class bidirectional_carbon_fiber(solid):

    """ Physical Constants of a Specific to Bi-Directional Carbon Fiber"""

    def __defaults__(self):

        self.ultimate_tensile_strength        = 600e6    *Units.Pa
        self.ultimate_shear_strength          = 90e6     *Units.Pa
        self.ultimate_bearing_strength        = 600e6    *Units.Pa
        self.yield_tensile_strength           = 600e6    *Units.Pa
        self.yield_shear_strength             = 90e6     *Units.Pa
        self.yield_bearing_strength           = 600e6    *Units.Pa
        self.minimum_gage_thickness           = 420e-6   *Units.m
        self.density                          = 1600.     *(Units.kg)/((Units.m)**3)
