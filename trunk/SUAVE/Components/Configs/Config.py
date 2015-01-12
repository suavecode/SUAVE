

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Structure import DiffedData

from copy import deepcopy

# ----------------------------------------------------------------------
#  Config
# ----------------------------------------------------------------------

class Config(DiffedData):
    """ SUAVE.Components.Config()
    """
    
    def __defaults__(self):
        self.tag    = 'config'
        

# ----------------------------------------------------------------------
#  Config Container
# ----------------------------------------------------------------------

class Container(DiffedData.Container):
    """ SUAVE.Components.Config.Container()
    """
    pass

# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Config.Container = Container
