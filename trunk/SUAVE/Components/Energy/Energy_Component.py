# ------------------------------------------------------------
#  Imports
# ------------------------------------------------------------

from SUAVE.Core import (
Data, Container, Data_Exception, Data_Warning,
)
from SUAVE.Components import Physical_Component

# ----------------------------------------------------------------------
#  Energy Component Class
# ----------------------------------------------------------------------

class Energy_Component(Physical_Component):
    def __defaults__(self):
        
        # function handles for input
        self.inputs  = Data()
        
        # function handles for output
        self.outputs = Data()
        
        return