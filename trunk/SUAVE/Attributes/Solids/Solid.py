## @ingroup Attributes-Solids

# Solid.py
#
# Created: Jul, 2017, J. Smart
# Modified: Apr, 2018, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Data, Units

#-------------------------------------------------------------------------------
# Solid Data Class
#-------------------------------------------------------------------------------

## @ingroup Attributes-Solids
class Solid(Data):
    """ Default Template for Solid Attribute Classes
    
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

    def __defaults__(self):
        """Default Instantiation of Physical Property Values
        
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

        self.ultimate_tensile_strength  = None
        self.ultimate_shear_strength    = None
        self.ultimate_bearing_strength  = None
        self.yield_tensile_strength     = None
        self.yield_shear_strength       = None
        self.yield_bearing_strength     = None
        self.minimum_gage_thickness     = None
        self.density                    = None
