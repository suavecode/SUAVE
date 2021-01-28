## @ingroup Attributes-Solids

# Aluminum.py
#
# Created: Jul, 2017, J. Smart
# Modified: Apr, 2018, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from .Solid import Solid
from SUAVE.Core import Units

#-------------------------------------------------------------------------------
# Aluminum 6061-T6 Solid Class
#-------------------------------------------------------------------------------

## @ingroup Attributes-Solid
class Aluminum(Solid):

    """ Physical Constants Specific to 6061-T6 Aluminum
    
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

        self.ultimate_tensile_strength  = 310e6 * Units.Pa
        self.ultimate_shear_strength    = 206e6 * Units.Pa
        self.ultimate_bearing_strength  = 607e6 * Units.Pa
        self.yield_tensile_strength     = 276e6 * Units.Pa
        self.yield_shear_strength       = 206e6 * Units.Pa
        self.yield_bearing_strength     = 386e6 * Units.Pa
        self.minimum_gage_thickness     = 0.0   * Units.m
        self.density                    = 2700. * Units['kg/(m**3)']
