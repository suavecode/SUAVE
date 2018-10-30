## @ingroup Components-Payloads
# Person.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Payload import Payload

# ----------------------------------------------------------------------
#  Person Data Class
# ----------------------------------------------------------------------
## @ingroup Components-Payloads
class Person(Payload):
    """A class representing a person.
    
    Assumptions:
    None
    
    Source:
    N/A
    """      
    def __defaults__(self):
        """This sets the default values for a person.

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
        self.tag = 'Person'
        self.mass_properties.mass = 90.718474     # kg, = 200 lb
