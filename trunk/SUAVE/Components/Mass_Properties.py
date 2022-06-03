## @ingroup Components
# Mass_Properties.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from jax.tree_util import register_pytree_node_class

import numpy as np

# ----------------------------------------------------------------------
#  Mass Properties
# ----------------------------------------------------------------------

## @ingroup Components
@register_pytree_node_class
class Mass_Properties(Data):
    """ Mass properties for a physical component
        
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
        
        self.mass   = 0.0
        self.volume = 0.0
        self.center_of_gravity = np.array([[0.0,0.0,0.0]])
        
        self.moments_of_inertia = Data()
        self.moments_of_inertia.center = np.array([0.0,0.0,0.0])
        self.moments_of_inertia.tensor = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])