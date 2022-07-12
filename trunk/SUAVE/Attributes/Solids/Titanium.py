## @ingroup Attributes-Solids

# Titanum.py
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
class Titanium(Solid):
    """ Physical Constants Specific to Grade 5 Ti-6AL-4V Alloy, Annealed

    Assumptions:
    None

    Source:
    ASM, Inc.
    https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=mtp641

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

        self.ultimate_tensile_strength  = 950e6 * Units.Pa
        self.ultimate_shear_strength    = 550e6 * Units.Pa
        self.ultimate_bearing_strength  = 1860e6 * Units.Pa
        self.yield_tensile_strength     = 880e6 * Units.Pa
        self.yield_shear_strength       = 550e6 * Units.Pa
        self.yield_bearing_strength     = 1480e6 * Units.Pa
        self.minimum_gage_thickness     = 0.0 * Units.m
        self.density                    = 4430. * Units['kg/(m**3)']
