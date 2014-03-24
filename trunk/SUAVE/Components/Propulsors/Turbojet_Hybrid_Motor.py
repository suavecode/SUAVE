""" Turbojet_Hybrid_Motor.py: Turbojet 1D gasdynamic engine model with motor """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure import Data
from Propulsor import Propulsor

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Turbojet_Hybrid_Motor(Propulsor):

    """ A Turbojet cycle Propulsor with motor and optional afterburning """

    def __defaults__(self):      
        raise NotImplementedError
        
    def initialize(self):
        raise NotImplementedError

    def __call__(self,eta,segment):
        raise NotImplementedError
