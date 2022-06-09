## @ingroup Analyses-Mission-Segments-Conditions
# Unknowns.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .Conditions import Conditions
from jax.tree_util import register_pytree_node_class

# ----------------------------------------------------------------------
#  Unknowns
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Conditions
@register_pytree_node_class
class Unknowns(Conditions):
    """ Creates the data structure for the unknowns that solved in a mission
    
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
        self.tag = 'unknowns'