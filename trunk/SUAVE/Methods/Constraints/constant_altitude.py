""" Constraints.py: Functions defining the constraints on dynamics """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure import Data

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

class constant_altitude(Data):

    """  dzdt = General2DOF(self,t,z): first-order ODE system for general 2D flight
    
         Inputs:    t = integration time (s)                                        (required)  (float)
                    z = vector of 1st-order ODE values                              (required)  (floats)
                    self.config = Vehicle.Configuration instance (supporting data)  (required)  (class)
  
         Outputs:   dzdt = time derivatives of equation system                      (floats)

    """
    def __defaults__(self):
        self.altitude = 0.0

    def __call__(self,segment):
        return (state.vectors.r[:,-1] - self.altitude)/self.altitude
