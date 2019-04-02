## @ingroup Attributes-Solids

# Aluminum_Rib.py
#
# Created: Jul, 2017, J. Smart
# Modified: Apr, 2018, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from .Solid import Solid
from .Aluminum import Aluminum
from SUAVE.Core import Units

#-------------------------------------------------------------------------------
# Aluminim Component Material Property Data Class
#-------------------------------------------------------------------------------

## @ingroup Attributes-Solids
class Aluminum_Rib(Aluminum):
    """ Physical Constants Specific to 6061-T6 Aluminum Ribs
    
    Assumptions:
    Limit of machining capability for precision components
    
    Source:
    None
    
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


        self.minimum_gage_thickness = 1.5e-3   * Units.m
        self.minimum_width          = 25.4e-3  * Units.m

