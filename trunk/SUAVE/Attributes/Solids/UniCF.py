## @ingroup Attributes-Solids

# UniCF.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Uni-Directional Carbon Fiber Solid Class
#-------------------------------------------------------------------------------

class UniCF(Solid):

    """ Physical Constants Specific to Uni-Directional Carbon Fiber"""

    def __defaults__(self):

        self.ultimate_tensile_strength        = 1500e6   *Units.Pa
        self.ultimate_shear_strength          = 70e6     *Units.Pa
        self.ultimate_bearing_strength        = 1500e6   *Units.Pa
        self.yield_tensile_strength           = 1500e6   *Units.Pa
        self.yield_shear_strength             = 70e6     *Units.Pa
        self.yield_bearing_strength           = 1500e6   *Units.Pa
        self.minimum_gage_thickness           = 420e-6   *Units.m
        self.density                          = 1600     *(Units.kg)/((Units.m)**3)
