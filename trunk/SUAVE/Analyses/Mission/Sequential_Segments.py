## @ingroup Analyses-Mission
# Sequential_Segments.py
#
# Created:  
# Modified: Feb 2016, A. Wendorff
#           Oct 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Methods import Missions as Methods
from .Mission import Mission
from jax.tree_util import register_pytree_node_class

# ----------------------------------------------------------------------
#   Class
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission
@register_pytree_node_class
class Sequential_Segments(Mission):
    """ Solves each segment one at time
    
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
        
        self.tag = 'mission'
        
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize
        # --------------------------------------------------------------
        self.process.initialize = Methods.Segments.Common.Sub_Segments.expand_sub_segments

        # --------------------------------------------------------------
        #   Converge
        # --------------------------------------------------------------
        self.process.converge = Methods.Segments.Common.Sub_Segments.sequential_sub_segments
        
        # --------------------------------------------------------------
        #   Iterate
        # --------------------------------------------------------------        
        del self.process.iterate


        return