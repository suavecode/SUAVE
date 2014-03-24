""" Turbojet_Hybrid_Motor_Fuel_Cell.py: Turbojet 1D gasdynamic engine model with motor and fuel cells """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure import Data
from Propulsor import Propulsor

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Turbojet_Hybrid_Motor_Fuel_Cell(Propulsor):

    """ A Turbojet cycle Propulsor with motor, fuell cells, and optional afterburning """

    def __defaults__(self):      
        raise NotImplementedError
        
    def initialize(self):
        raise NotImplementedError

    def __call__(self,eta,segment):
        raise NotImplementedError
