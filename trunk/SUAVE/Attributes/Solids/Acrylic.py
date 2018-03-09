## @ingroup Attributs-Solids

# Acrylic.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import solid
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Acrylic Solid Class
#-------------------------------------------------------------------------------

class acrylic(solid):

    """ Physical Constants Specific to Polymethyl Methacrylate"""

    def __defaults__(self):

        self.ultimate_tensile_strength        = 75e6          *Units.Pa
        self.ultimate_shear_strength          = 55.2e6        *Units.Pa
        self.ultimate_bearing_strength        = 0.0           *Units.Pa
        self.yield_tensile_strength           = 75e6          *Units.Pa
        self.yield_shear_strength             = 55.2e6        *Units.Pa
        self.yield_bearing_strength           = 0.0           *Units.Pa
        self.minimum_gage_thickness           = 3.175e-3      *Units.m
        self.density                          = 1180          *(Units.kg)/((Units.m)**3)
