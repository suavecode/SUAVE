## @ingroup Attributes-Solids

# Paint.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Paint and/or Vinyl Surface Convering Solid Class
#-------------------------------------------------------------------------------

class paint(solid):

    """ Physical Constants Specific to Paint and/or Vinyl Surface Covering"""

    def __defaults__(self):

        self.ultimate_tensile_strength        = 0.0       *Units.Pa
        self.ultimate_shear_strength          = 0.0       *Units.Pa
        self.ultimate_bearing_strength        = 0.0       *Units.Pa
        self.yield_tensile_strength           = 0.0       *Units.Pa
        self.yield_shear_strength             = 0.0       *Units.Pa
        self.yield_bearing_strength           = 0.0       *Units.Pa
        self.minimum_gage_thickness           = 150e-6    *Units.m
        self.density                          = 1800       *(Units.kg)/((Units.m)**3)
