## @ingroup Attributes-Solids

# Rib.py
#
# Created: Jul 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from Solid import Solid
from Aluminum import Aluminum
from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Solid Data Class
#-------------------------------------------------------------------------------

class Rib(Aluminum):

    """ Physical Constants of an Aluminum 6061-T6 Rib"""

    def __defaults__(self):


        self.minimumGageThickness           = 1.5e-3   *Units.m
        self.minWidth                       = 25.4e-3  *Units.m

