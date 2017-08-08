## @ingroup Components-Payloads
# Payload.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Components import Physical_Component


# ----------------------------------------------------------------------
#  Payload Base Class
# ----------------------------------------------------------------------
## @ingroup Components-Payloads    
class Payload(Physical_Component):
    """A class representing a payload.
    
    Assumptions:
    None
    
    Source:
    N/A
    """      
    def __defaults__(self):
        """This sets the default tag for a payload.

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
        self.tag = 'Payload'
        
## @ingroup Components-Payloads  
class Container(Physical_Component.Container):
    """The container used for payloads. No additional functionality.
    
    Assumptions:
    None
    
    Source:
    N/A
    """      
    pass

Payload.Container = Container

