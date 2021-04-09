## @ingroup Attributes-Solids

# Unidirectional_Carbon_Fiber.py
#
# Created: Jul, 2017, J. Smart
# Modified: Apr, 2018, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from .Solid import Solid
from SUAVE.Core import Units

#-------------------------------------------------------------------------------
# Uni-Directional Carbon Fiber Solid Class
#-------------------------------------------------------------------------------

## @ingroup Attributes-Solids
class Unidirectional_Carbon_Fiber(Solid):

    """ Physical Constants Specific to Unidirectional Carbon Fiber
    
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

        self.ultimate_tensile_strength  = 1500e6    * Units.Pa
        self.ultimate_shear_strength    = 70e6      * Units.Pa
        self.ultimate_bearing_strength  = 1500e6    * Units.Pa
        self.yield_tensile_strength     = 1500e6    * Units.Pa
        self.yield_shear_strength       = 70e6      * Units.Pa
        self.yield_bearing_strength     = 1500e6    * Units.Pa
        self.minimum_gage_thickness     = 420e-6    * Units.m
        self.density                    = 1600.     * Units['kg/(m**3)']
