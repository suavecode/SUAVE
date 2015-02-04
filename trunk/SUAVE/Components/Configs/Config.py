

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Core import Diffed_Data

from copy import deepcopy

# ----------------------------------------------------------------------
#  Config
# ----------------------------------------------------------------------

class Config(Diffed_Data):
    """ SUAVE.Components.Config()
    """
    
    def __defaults__(self):
        self.tag    = 'config'
        

# ----------------------------------------------------------------------
#  Config Container
# ----------------------------------------------------------------------

class Container(Diffed_Data.Container):
    """ SUAVE.Components.Config.Container()
    """
    pass

# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Config.Container = Container
