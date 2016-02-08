# Config.py
#
# Created:  Oct 2014, T. Lukacyzk
# Modified: Jan 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Diffed_Data

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
