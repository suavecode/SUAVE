""" Turbofan_Hybrid_Motor_Fuel_Cell.py: Turbofan 1D gasdynamic engine model with motor and fuel cells """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data
from Propulsor import Propulsor

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Turbofan_Hybrid_Motor_Fuel_Cell(Propulsor):

    """ A Turbofan cycle Propulsor with motor, fuel cellss, and optional afterburning """

    def __defaults__(self):      
        raise NotImplementedError
        
    def initialize(self):
        raise NotImplementedError

    def __call__(self,eta,segment):
        raise NotImplementedError