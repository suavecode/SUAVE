## @ingroup Attributes-Solids

# Solid.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Solid Data Class
#-------------------------------------------------------------------------------

class Solid(Data):

    """ Physical Constants of a Solid"""

    def __defaults__(self):

        self.ultimate_tensile_strength        = 0.0   *Units.Pa
        self.ultimate_shear_strength          = 0.0   *Units.Pa
        self.ultimate_bearing_strength        = 0.0   *Units.Pa
        self.yield_tensile_strength           = 0.0   *Units.Pa
        self.yield_shear_strength             = 0.0   *Units.Pa
        self.yield_bearing_strength           = 0.0   *Units.Pa
        self.minimum_gage_thickness           = 0.0   *Units.m
        self.density                          = 0.0   *(Units.kg)/((Units.m)**3) 
