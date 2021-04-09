## @ingroup Attributes-Solids

# Steel.py
#
# Created: Jul, 2017, J. Smart
# Modified: Apr, 2018, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from .Solid import Solid
from SUAVE.Core import Units

#-------------------------------------------------------------------------------
# AISI 4340 Steel Solid Class
#-------------------------------------------------------------------------------

## @ingroup Attributes-Solids
class Steel(Solid):
    """ Physical Constants Specific to AISI 4340 Steel
    
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

        self.ultimate_tensile_strength  = 1110e6    * Units.Pa
        self.ultimate_shear_strength    = 825e6     * Units.Pa
        self.ultimate_bearing_strength  = 1110e6    * Units.Pa
        self.yield_tensile_strength     = 710e6     * Units.Pa
        self.yield_shear_strength       = 410e6     * Units.Pa
        self.yield_bearing_strength     = 710e6     * Units.Pa
        self.minimum_gage_thickness     = 0.0       * Units.m
        self.density                    = 7850.     * Units['kg/(m**3)']
