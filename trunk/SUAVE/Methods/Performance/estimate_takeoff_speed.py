""" estimate_takeoff_speed.py: ... """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import math
import copy

from SUAVE.Core            import Data
from SUAVE.Attributes.Results   import Result, Segment
# from SUAVE.Methods.Utilities    import chebyshev_data, pseudospectral

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def estimate_takeoff_speed(config,segment,tol=1e-6,guess=0.0):

    # initilize
    N = segment.options.Npoints; m = 5; z = np.zeros(m)
    
    state = State(); 
    m0 = config.Mass_Properties.takeoff                                 # kg
    g0 = segment.planet.sea_level_gravity                          # m/s^2
    z[2] = segment.airport.altitude                                 # m    
    z[4] = m0                                                       # kg
    state.alpha = np.radians(segment.climb_angle)                   # rad

    # estimate liftoff speed in this configuration
    dV = 1.0; V_lo = guess
    while dV > tol:        
       
        z[1] = V_lo
        state.compute_state(z,config,segment,["no vectors", "constant altitude"])
        V_lo_new = np.sqrt(2*(m0*g0 - state.T*np.sin(state.gamma))/(state.CL*state.rho*config.S))
        dV = np.abs(V_lo_new - V_lo)
        # print "dV = ", dV
        V_lo = V_lo_new

    return state
