""" Rotor.py: Rotor propulsion model """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure import Data
from Propulsor import Propulsor

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Rotor(Propulsor):

    """ A Rotor Propulsor """

    def __defaults__(self):      
        raise NotImplementedError
        
    def initialize(self):
        raise NotImplementedError

    def __call__(self,eta,segment):
        raise NotImplementedError