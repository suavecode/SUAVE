""" check_segment.py: ... """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import math
# import Units
import copy

from SUAVE.Structure            import Data
from SUAVE.Attributes.Results   import Result, Segment
# from Utilities                  import chebyshev_data, pseudospectral

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def check_segment(config,segment,ICs):

    check = Data()
    check.errors = []; check.warnings = []; check.error = False; check.warning = False

    if segment.type.lower() == 'climb':                     # Climb segment

        name = "Climb Segment: "
        
        # check variables existence & values
        try: 
            segment.altitude
        except NameError:
            check.errors.append(name + "altitude value not defined; please check inputs"); check.error = True
        else:
            if segment.altitude <= 0.0:
                check.errors.append(name + "final altitude is <= 0; please check inputs"); check.error = True
        
        try: 
            segment.atmosphere
        except NameError:
            check.errors.append(name + "atmosphere not defined; please check inputs"); check.error = True
        else:
            pass

        try: 
            config.Function.Propulsion
        except NameError:
            check.errors.append(name + "no propulsion function defined; please check inputs"); check.error = True
        else:
            pass

        try: 
            config.Function.Aero
        except NameError:
            check.errors.append(name + "no aerodynamic function defined; please check inputs"); check.error = True
        else:
            pass

        # check ICs
        if not check.error:
            if ICs[1] <= 0.0:
                check.errors.append(name + "vehicle is at rest flying backward"); check.error = True
            if ICs[2] >= segment.altitude:
                check.errors.append(name + "vehicle is above specified final altitude for climb segment"); check.error = True
            if ICs[3] < 0.0:
                check.warnings.append(name + "vehicle is falling at beginning of climb segment"); check.warning = True
     
        # check physics
        if not check.error:

            # start point
            state = State(); z = ICs
            state.ComputeState(z,segment,config)
   
            L, D = config.Functions.Aero(state)*state.q*config.S            # N
            T, mdot = config.Functions.Propulsion(state)                    # N, kg/s

            # common trig terms
            sin_gamma_alpha = np.sin(state.gamma - state.alpha)
            cos_gamma_alpha = np.cos(state.gamma - state.alpha)
    
            # drag terms
            Dv = np.zeros(2)
            Dv[0] = -D*cos_gamma_alpha                              # x-component 
            Dv[1] = -D*sin_gamma_alpha                              # y-component

            # lift terms
            Lv = np.zeros(2)
            Lv[0] = L*sin_gamma_alpha                               # x-component 
            Lv[1] = L*cos_gamma_alpha                               # y-component

            # thurst terms
            Tv = np.zeros(2)
            Tv[0] = T*np.cos(np.radians(state.gamma + state.delta)) # x-component
            Tv[1] = T*np.sin(np.radians(state.gamma + state.delta)) # y-component

            d2xdt2 = (Dv[0] + Lv[0] + Tv[0])/state.m
            d2ydt2 = (Dv[1] + Lv[1] + Tv[1])/state.m - state.g

    return err, warn, reason