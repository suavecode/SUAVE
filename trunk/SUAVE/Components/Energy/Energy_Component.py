## @ingroup Energy
# __init__.py
# 
# Created:  Aug 2014, E. Botero
# Modified: Feb 2016, T. MacDonald

# ------------------------------------------------------------
#  Imports
# ------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Components import Physical_Component

# ----------------------------------------------------------------------
#  Energy Component Class
# ----------------------------------------------------------------------
## @ingroup Energy
class Energy_Component(Physical_Component):
    """A class representing an energy component.
    
    Assumptions:
    None
    
    Source:
    N/A
    """      
    def __defaults__(self):
        """This sets the default inputs and outputs data structure.

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
        # function handles for input
        self.inputs  = Data()
        
        # function handles for output
        self.outputs = Data()
        
        return