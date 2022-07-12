## @ingroup Attributes-Solids

# Magnesium.py
#
# Created: Jul, 2022, J. Smart
# Modified:

# -------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------

from .Solid import Solid
from SUAVE.Core import Units


# -------------------------------------------------------------------------------
# Aluminum 6061-T6 Solid Class
# -------------------------------------------------------------------------------

## @ingroup Attributes-Solid
class Magnesium(Solid):
    """ Physical Constants Specific to RZ5 Magnesium Alloy per BS 2L.128

    Assumptions:
    None

    Source:
    MatWeb
    https://www.matweb.com/search/datasheet.aspx?matguid=f473a6bbcabe4fd49199c2cef7205664

    Inputs:
    N/A

    Outputs:
    N/A

    Properties Used:
    None
    """

    def __defaults__(self):
        """Sets material properties at instantiation.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        N/A

        Outputs:
        N/A

        Properties Used:
        None
        """

        self.ultimate_tensile_strength  = 200e6 * Units.Pa
        self.ultimate_shear_strength    = 138e6 * Units.Pa
        self.ultimate_bearing_strength  = 330e6 * Units.Pa
        self.yield_tensile_strength     = 135e6 * Units.Pa
        self.yield_shear_strength       = 138e6 * Units.Pa
        self.yield_bearing_strength     = 130e6 * Units.Pa
        self.minimum_gage_thickness     = 0.0   * Units.m
        self.density                    = 1840. * Units['kg/(m**3)']
