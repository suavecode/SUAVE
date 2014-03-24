

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Container, Data_Exception, Data_Warning
from Component import Component


# ----------------------------------------------------------------------
#  Envelope 
# ----------------------------------------------------------------------

class Envelope(Component):
    def __defaults__(self):
        self.tag = 'Envelope'
        self.alpha_limit = 0.0
        self.cg_ctrl     = 0.0
        self.alt_vc      = 0.0
        self.alt_gust    = 0.0
        self.max_ceiling = 0.0
        self.mlafactor   = 0.0
        self.glafactor   = 0.0