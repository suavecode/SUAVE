## @ingroup Analyses-Mission-Segments-Conditions
# Residuals.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Conditions import Conditions

# ----------------------------------------------------------------------
#  Residuals
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Conditions
class Residuals(Conditions):
    """ Creates the data structure for the residuals that solved in a mission
    
        Assumptions:
        None
        
        Source:
        None
    """    
    
    def __defaults__(self):
        """This sets the default values.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """           
        self.tag = 'residuals'