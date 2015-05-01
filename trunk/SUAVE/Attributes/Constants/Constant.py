""" Constant.py: Physical constants class """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception, Data_Warning
from SUAVE.Core import Container as ContainerBase

# ----------------------------------------------------------------------
#  Constants 
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
    
    