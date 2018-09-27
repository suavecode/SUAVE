## @ingroup Components-Payloads
# Satellite.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald

""" SUAVE Vehicle container class 
    with database + input / output functionality 
"""


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Payload import Payload

# ----------------------------------------------------------------------
#  Sattelite Data Class
# ----------------------------------------------------------------------
## @ingroup Components-Payloads
class Satellite(Payload):
    """A class representing a satellite.
    
    Assumptions:
    None
    
    Source:
    N/A
    """          
    def __defaults__(self):
        """This sets the default tag for a satellite.

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
        self.tag = 'Satellite'