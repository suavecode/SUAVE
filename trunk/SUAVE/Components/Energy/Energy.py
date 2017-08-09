## @ingroup Energy
# __init__.py
# 
# Created:  Aug 2014, E. Botero
# Modified: Feb 2016, T. MacDonald

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


# ------------------------------------------------------------
#  Energy Component Classes
# ------------------------------------------------------------
## @ingroup Energy
class Component(Physical_Component):
    """A class representing an generic energy component.
    
    Assumptions:
    None
    
    Source:
    N/A
    """     
    def __defaults__(self):
        """This sets the default tag for a component.

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
        self.tag = 'Energy Component'
    
    def provide_power():
        """A stub for providing power.

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
    
## @ingroup Energy
class ComponentContainer(Physical_Component.Container):
    """A container for an energy component.
    
    Assumptions:
    None
    
    Source:
    N/A
    """     
    pass

# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------
Energy.Component = Component
Energy.Component.Container = ComponentContainer


