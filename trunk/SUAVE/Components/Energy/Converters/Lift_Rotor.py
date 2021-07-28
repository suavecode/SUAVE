## @ingroup Components-Energy-Converters
# Lift_Rotor.py
#
# Created:  July 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from .Rotor import Rotor

# ----------------------------------------------------------------------
#  Lift Rotor Class
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Converters
class Lift_Rotor(Rotor):
    """This is a lift rotor component.
    
    Assumptions:
    None
    
    Source:
    None
    """     
    def __defaults__(self):
        """This sets the default values for the component to function.
        
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

        self.tag                   = 'lift_rotor'
        self.use_2d_analysis       = True
        
        