## @ingroup Attributes-Solids

# Epoxy.py
#
# Created: Jul, 2017, J. Smart
# ModifiedL Apr, 2018, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from .Solid import Solid
from SUAVE.Core import Units

#-------------------------------------------------------------------------------
# Hardened Epoxy Resin Solid Class
#-------------------------------------------------------------------------------

## @ingroup Attributes-Solids
class Epoxy(Solid):
    """ Physical Constants Specific to Epoxy Resin
    
    Assumptions:
    None
    
    Source:
    MatWeb (Median of Mfg. Reported Values)
    
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

        self.ultimate_tensile_strength  = 0.0       * Units.Pa
        self.ultimate_shear_strength    = 0.0       * Units.Pa
        self.ultimate_bearing_strength  = 0.0       * Units.Pa
        self.yield_tensile_strength     = 0.0       * Units.Pa
        self.yield_shear_strength       = 0.0       * Units.Pa
        self.yield_bearing_strength     = 0.0       * Units.Pa
        self.minimum_gage_thickness     = 250e-6    * Units.m
        self.density                    = 1800.     * Units['kg/(m**3)']
