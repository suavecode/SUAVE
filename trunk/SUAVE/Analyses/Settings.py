# Settings.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
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