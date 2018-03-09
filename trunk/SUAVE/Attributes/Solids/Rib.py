## @ingroup Attributes-Solids

# Rib.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import solid
from Aluminum import aluminum
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Solid Data Class
#-------------------------------------------------------------------------------

class rib(aluminum):

    """ Physical Constants of an Aluminum 6061-T6 Rib"""

    def __defaults__(self):


        self.minimum_gage_thickness              = 1.5e-3   *Units.m
        self.minimum_width                       = 25.4e-3  *Units.m

