## @ingroup Energy
# __init__.py
# 
# Created:  Aug 2014, E. Botero
# Modified: Feb 2016, T. MacDonald
#           Oct 2017, E. Botero

# ------------------------------------------------------------
#  Imports
# ------------------------------------------------------------

from SUAVE.Components import Physical_Component

# ------------------------------------------------------------
#  The Home Energy Container Class
# ------------------------------------------------------------
## @ingroup Energy
class Energy(Physical_Component):
    """A class representing an energy component.
    
    Assumptions:
    None
    
    Source:
    N/A
    """     
    def __defaults__(self):
        """This sets the defaults. (Currently empty)

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
        pass