## @ingroup Components-Wings
# Stabilator.py
#
# Created:  Jul 2021, A. Blaufox
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# MARC imports
from .Horizontal_Tail    import Horizontal_Tail
from .All_Moving_Surface import All_Moving_Surface

# ----------------------------------------------------------------------
#  Attribute
# ----------------------------------------------------------------------

## @ingroup Components-Wings
class Stabilator(Horizontal_Tail, All_Moving_Surface):
    """ This class is used to define stabilators in MARC. Note that it 
    inherits from both Horizontal_Tail and All_Moving_Surface
    
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
        """This sets the default for stabilators in MARC.
        
        See All_Moving_Surface().__defaults__ and Wing().__defaults__ for 
        an explanation of attributes
    
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
        self.tag = 'stabilator'
        self.sign_duplicate        = 1.0
