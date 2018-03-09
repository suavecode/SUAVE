## @ingroup Attributes-Solids

# Epoxy.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Hardened Epoxy Resin Solid Class
#-------------------------------------------------------------------------------

class epoxy(solid):

    """ Physical Constants Specific to Hardened Expoy Resin"""

    def __defaults__(self):

        self.ultimate_tensile_strength        = 0.0      *Units.Pa
        self.ultimate_shear_strength          = 0.0      *Units.Pa
        self.ultimate_bearing_strength        = 0.0      *Units.Pa
        self.yield_tensile_strength           = 0.0      *Units.Pa
        self.yield_shear_strength             = 0.0      *Units.Pa
        self.yield_bearing_strength           = 0.0      *Units.Pa
        self.minimum_gage_thickness           = 250e-6   *Units.m
        self.density                          = 1800     *(Units.kg)/((Units.m)**3)
