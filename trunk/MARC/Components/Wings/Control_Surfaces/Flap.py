## @ingroup Components-Wings-Control_Surfaces
# Flap.py
#
# Created:  Jan 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# MARC imports
from MARC.Components.Wings.Control_Surfaces.Control_Surface import Control_Surface 

# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------
## @ingroup Components-Wings-Control_Surfaces
class Flap(Control_Surface):
    """This class is used to define slats in MARC

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
        """This sets the default for slats in MARC.
        
        see Control_Surface().__defaults__ for an explanation of attributes
    
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
        
        self.tag                   = 'flap'
        self.hinge_fraction        = 0.0
        self.sign_duplicate        = 1.0        
        
        pass
 