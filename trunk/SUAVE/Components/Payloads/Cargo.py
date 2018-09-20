## @ingroup Components-Payloads
# Cargo.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Payload import Payload

# ----------------------------------------------------------------------
#  Cargo Data Class
# ----------------------------------------------------------------------
## @ingroup Components-Payloads
class Cargo(Payload):
    """A class representing cargo.
    
    Assumptions:
    None
    
    Source:
    N/A
    """
    def __defaults__(self):
        """This sets the default tag for cargo.

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
        self.tag = 'Cargo'
