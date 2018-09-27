## @ingroup Attributes-Solids

# Carbon_Fiber_Honeycomb.py
#
# Created: Jul, 2017, J. Smart
# Modified: Apr, 2018, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from .Solid import Solid
from SUAVE.Core import Units

#-------------------------------------------------------------------------------
# Carbon Fiber Honeycomb Core Solid Class
#-------------------------------------------------------------------------------

## @ingroup Attributes-Solids
class Carbon_Fiber_Honeycomb(Solid):
    """ Physical Constants Specific to Carbon Fiber Honeycomb Core Material
    
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

        self.ultimate_tensile_strength  = 1e6       * Units.Pa
        self.ultimate_shear_strength    = 1e6       * Units.Pa
        self.ultimate_bearing_strength  = 1e6       * Units.Pa
        self.yield_tensile_strength     = 1e6       * Units.Pa
        self.yield_shear_strength       = 1e6       * Units.Pa
        self.yield_bearing_strength     = 1e6       * Units.Pa
        self.minimum_gage_thickness     = 6.5e-3    * Units.m
        self.density                    = 55.       * Units['kg/(m**3)']
