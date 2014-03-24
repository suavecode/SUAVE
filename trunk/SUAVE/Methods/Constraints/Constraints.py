""" Constraints.py: Functions defining the constraints on dynamics """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure import Data

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def horizontal_force(segment,F_mg=0.0):

    m = segment.m
    m[m < segment.config.Mass_Props.m_empty] = segment.config.Mass_Props.m_empty

    return segment.vectors.Ftot[:,0]/(m*segment.g) - F_mg

def vertical_force(segment,F_mg=0.0):

    m = segment.m
    m[m < segment.config.Mass_Props.m_empty] = segment.config.Mass_Props.m_empty

    return segment.vectors.Ftot[:,2]/(m*segment.g) - 1.0 - F_mg

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

class constant_mach(Data):

    """  dzdt = General2DOF(self,t,z): first-order ODE system for general 2D flight
    
         Inputs:    t = integration time (s)                                        (required)  (float)
                    z = vector of 1st-order ODE values                              (required)  (floats)
                    self.config = Vehicle.Configuration instance (supporting data)  (required)  (class)
  
         Outputs:   dzdt = time derivatives of equation system                      (floats)

    """
    def __defaults__(self):
        self.Minf = 0.0
    
    def __call__(self,z,u,eta,D,I,state):
        return state.M - self.Minf

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

