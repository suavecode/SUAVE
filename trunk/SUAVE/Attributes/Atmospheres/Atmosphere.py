""" Atmosphere.py: Constant-property atmopshere model """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# classes
import numpy as np
from SUAVE.Attributes.Gases import Air
from SUAVE.Attributes.Constants import Constant, Composition
from SUAVE.Core import Data, Data_Exception, Data_Warning

# other
from numpy import sqrt, exp, abs

# ----------------------------------------------------------------------
#  Atmosphere Data Class
# ----------------------------------------------------------------------

class Atmosphere(Constant):

    """ SUAVE.Attributes.Atmospheres.Atmosphere
    """

    def __defaults__(self):
        self.tag = 'Constant-property atmopshere'
        self.composition           = Data()
        self.composition.gas       = 1.0
