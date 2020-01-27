## @ingroup Components-Wings-Control_Surfaces
# Flap.py
#
# Created:  Jan 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Components.Wings.Control_Surfaces.Control_Surface import Control_Surface 

# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------
## @ingroup Components-Wings-Control_Surfaces
class Flap(Control_Surface):
    """This class is used to define slats in SUAVE

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    None

    Outputs:
    None

    Properties Used:
    N/A
    """ 
    def __defaults__(self):
        """This sets the default for slats in SUAVE.
    
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """ 
        
        self.tag      = 'flap'
        
        pass
 