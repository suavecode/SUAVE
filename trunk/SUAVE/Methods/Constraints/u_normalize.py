""" Constraints.py: Functions defining the constraints on dynamics """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


class u_normalize(Data):

    """  dzdt = General2DOF(self,t,z): first-order ODE system for general 2D flight
    
         Inputs:    t = integration time (s)                                        (required)  (float)
                    z = vector of 1st-order ODE values                              (required)  (floats)
                    self.config = Vehicle.Configuration instance (supporting data)  (required)  (class)
  
         Outputs:   dzdt = time derivatives of equation system                      (floats)

    """
    def __defaults__(self):
        self.u_norm = 1.0

    def __call__(self,z,u,eta,D,I,state):
        return np.sqrt(np.sum(u**2,axis=1)) - self.unorm