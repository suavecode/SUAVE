#Constant.py

# Created:  Mar, 2014, SUAVE Team
# Modified: Jan, 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Core import Container as ContainerBase

# ----------------------------------------------------------------------
#  Constant Data Class
# ----------------------------------------------------------------------

class Constant(Data):
    """ Constant Base Class """
    def __defaults__(self):
        pass

class Container(ContainerBase):
    pass

# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

Constant.Container = Container    
    
    