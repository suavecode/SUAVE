

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Core import Container as ContainerBase


# ----------------------------------------------------------------------
#  Component
# ----------------------------------------------------------------------

class Component(Data):
    """ SUAVE.Components.Component()
        the base component class
    """
    def __defaults__(self):
        self.tag    = 'Component'
        self.origin = [0.0,0.0,0.0]
    
    
# ----------------------------------------------------------------------
#  Component Container
# ----------------------------------------------------------------------

class Container(ContainerBase):
    """ SUAVE.Components.Component.Container()
        the base component container class
    """
    pass


# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Component.Container = Container