""" initialize_takeoff.py: ... """

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

def initialize_takeoff(config,segment):
    
    # initialize
    takeoff = None; N = segment.options.Npoints; m = 5

    # estimate liftoff speed in this configuration
    liftoff_state = estimate_takeoff_speed(config,segment)   
    
    # check for T > D at liftoff
    if liftoff_state.D > liftoff_state.T:
        print "Vehcile cannot take off: Drag > Trust at liftoff speed in this configuration."
        print "Adding phantom lift to compensate."
        
        CL_phantom = 0.0
        while dV > tol:        
       
            z[1] = V_lo
            state.compute_state(z,config,segment,["no vectors", "constant altitude"])
            V_lo_new = np.sqrt(2*(m0*g0 - state.T*np.sin(state.gamma))/(state.CL*state.rho*config.S))
            dV = np.abs(V_lo_new - V_lo)
            print "dV = ", dV
            V_lo = V_lo_new 

    else:
        T_lo = state.T; mdot_lo = state.mdot

    # get average properties over takeoff
    z[1] = 0.0; state.compute_state(z,config,segment,["no vectors", "constant altitude"])
    T0 = state.T; mdot0 = state.mdot
    T = (T0 + T_lo)/2; mdot = (mdot0 + mdot_lo)/2

    # estimate time to liftoff
    print state.CD, state.CL
    print 0.5*state.CD*state.rho*config.S*V_lo**2
    print state.T
    C = 0.5*state.CD*state.rho*config.S
    k1 = np.sqrt(C*T)/mdot; k2 = np.sqrt(C/T)
    p = k1*np.arctanh(k2*V_lo)
    
    tf = (m0/mdot)*(1 - np.exp(-p))

    # estimate state variables 
    t_cheb, D, I = chebyshev_data(N)
    t = t_cheb*tf
    z = np.zeros((N,m))
    z[:,1] = k2*np.tanh(k1*np.log(m0/(m0 - mdot*t)))                # Vx
    z[:,2] = segment.airport.altitude*np.ones(N)      # y

    # integrate Vx for x(t)
    z[:,0] = np.dot(I,z[:,1])                                       # x

    # pack up problem class
    takeoff = Problem()
    takeoff.tag = "Takeoff"
    takeoff.Npoints = N

    takeoff.f = EquationsOfMotion.ground_1DOF
    takeoff.FC = EquationsOfMotion.takeoff_speed
    takeoff.scale.z = zscale   
    takeoff.scale.t = tf
    takeoff.scale.F = m0*g0
    takeoff.t = t
    takeoff.z0 = z
    takeoff.t0 = 0.0                # initial time
    takeoff.tf = tf                 # final time guess
    takeoff.config = config

    return takeoff
