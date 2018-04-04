## @ingroup Attributes-Solids

# Steel.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# AISI 4340 Steel Solid Class
#-------------------------------------------------------------------------------

class steel(solid):

    """ Physical Constants Specific to AISI 4340 Steel"""

    def __defaults__(self):

        self.ultimate_tensile_strength        = 1110e6    *Units.Pa
        self.ultimate_shear_strength          = 825e6     *Units.Pa
        self.ultimate_bearing_strength        = 1110e6    *Units.Pa
        self.yield_tensile_strength           = 710e6     *Units.Pa
        self.yield_shear_strength             = 410e6     *Units.Pa
        self.yield_bearing_strength           = 710e6     *Units.Pa
        self.minimum_gage_thickness           = 0.0       *Units.m
        self.density                          = 7850.      *(Units.kg)/((Units.m)**3)
