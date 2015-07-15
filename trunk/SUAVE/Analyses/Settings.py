

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Core import Container as ContainerBase


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Settings(Data):
    """ SUAVE.Analyses.Settings()
    """
    def __defaults__(self):
        self.tag    = 'settings'
        
        self.verbose_process = False
        

# ----------------------------------------------------------------------
#  Config Container
# ----------------------------------------------------------------------

class Container(ContainerBase):
    """ SUAVE.Analyses.Settings.Container()
    """
    

# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Settings.Container = Container