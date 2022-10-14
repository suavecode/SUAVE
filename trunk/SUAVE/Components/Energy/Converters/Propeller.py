## @ingroup Components-Energy-Converters
# Propeller.py
#
# Created:  July 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from .Rotor import Rotor
from jax.tree_util import register_pytree_node_class

# ----------------------------------------------------------------------
#  Propeller Class
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Converters
@register_pytree_node_class
class Propeller(Rotor):
    """This is a propeller component, and is a sub-class of rotor.
    
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

        self.tag                       = 'propeller'
        self.orientation_euler_angles  = [0.,0.,0.] # This is X-direction thrust in vehicle frame
        self.variable_pitch            = False
        
