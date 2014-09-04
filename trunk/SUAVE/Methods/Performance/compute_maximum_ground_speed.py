""" compute_maximum_ground_speed.py: ... """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import math
import SUAVE.Methods.Units
import copy

from SUAVE.Structure            import Data
from SUAVE.Attributes.Results   import Result, Segment
# from SUAVE.Methods.Utilities    import chebyshev_data, pseudospectral

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def compute_maximum_ground_speed(config,segment,tol=1e-6,guess=0.0):

    # initilize
    N = segment.options.Npoints; m = 5 
    
    state = State(); z = np.zeros(m)
    z[2] = segment.airport.altitude                                 # m    
    z[4] = config.mass_properties.takeoff                           # kg
    state.alpha = 0.0                                               # rad

    # estimate liftoff speed in this configuration
    dV = 1.0; V = guess
    while dV > tol:        
       
        z[1] = V
        state.compute_state(z,config,segment,["no vectors", "constant altitude"])
        V_new = np.sqrt(2*state.T*np.cos(state.delta)/(state.CD*state.rho*config.S))
        dV = np.abs(V_new - V)
        V = V_new

    return state