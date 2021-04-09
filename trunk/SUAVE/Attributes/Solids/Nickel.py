## @ingroup Attributes-Solids

# Nickel.py
#
# Created: Jul, 2017, J. Smart
# Modified: Apr, 2018, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from .Solid import Solid
from SUAVE.Core import Units

#-------------------------------------------------------------------------------
# Cold Rolled Nickel/Cobalt Chromoly Alloy Solid Class
#-------------------------------------------------------------------------------

## @ingroup Attributes-Solids
class Nickel(Solid):
    """ Physical Constants Specific to Nickel/Cobalt Chromoly Alloy
    
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

        self.ultimate_tensile_strength  = 1830e6    * Units.Pa
        self.ultimate_shear_strength    = 1050e6    * Units.Pa
        self.ultimate_bearing_strength  = 1830e6    * Units.Pa
        self.yield_tensile_strength     = 1550e6    * Units.Pa
        self.yield_shear_strength       = 1050e6    * Units.Pa
        self.yield_bearing_strength     = 1550e6    * Units.Pa
        self.minimum_gage_thickness     = 0.0       * Units.m
        self.density                    = 8430.     * Units['kg/(m**3)']
