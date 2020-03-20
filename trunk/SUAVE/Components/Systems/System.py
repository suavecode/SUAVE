## @ingroup Components-Systems
# System.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald
#           Oct 2017, E. Botero
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Components import Physical_Component

# ----------------------------------------------------------------------
#  Payload Base Class
# ----------------------------------------------------------------------
        
## @ingroup Components-Systems
class System(Physical_Component):
    """A class representing an aircraft system/systems.
    
    Assumptions:
    None
    
    Source:
    N/A
    """  
    def __defaults__(self): 
        """ This sets the default values for the system.
        
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
        
        self.tag             = 'System'  
        self.control         = None
        self.accessories     = None