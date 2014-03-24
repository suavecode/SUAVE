""" evaluate_segment.py: ... """

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
        
def evaluate_segment(config,segment,ICs,solver):

    problem = Problem()
    options = Options()                                # ODE integration options
    problem.config = config
    problem.segment = segment
    err = False
    
    if segment.type.lower() == 'climb':             # Climb segment
        
        problem.tag = "Climb"
        problem.f = EquationsOfMotion.General2DOF

        # initial conditions
        problem.z0 = ICs

        if solver.lower() == "RK45":

            # boundary conditions
            problem.zmin = np.zeros(5)
            problem.zmin[0] = None                              # x: no minimum
            problem.zmin[1] = None                               # Vx: must be > 0
            problem.zmin[2] = 0.0                               # y: y > 0 (must climb)
            problem.zmin[3] = None                               # Vy: Vy > 0 (must climb)
            problem.zmin[4] = config.Mass_Props.m_flight_min     # m: mimumum mass (out of fuel)
        
            problem.zmax = np.zeros(5)
            problem.zmax[0] = None                              # x: no maximum
            problem.zmax[1] = None                              # Vx: liftoff when Vx >= V_liftoff
            problem.zmax[2] = segment.altitude[1]               # y: climb target
            problem.zmax[3] = None                              # Vy: no maximum 
            problem.zmax[4] = None                              # m: no maximum

            # check if this problem is well-posed 
            #y0 = segment.altitude[0]
            #p, T, rho, a, mew = segment.atmosphere.ComputeValues(y0,"all")

            # estimate time step
            problem.h0 = 0.01

        elif solver.lower() == "PS":

            # final conditions   
            problem.zf = np.zeros(5)
            problem.zf[0] = None                              # x: no maximum
            problem.zf[1] = None                              # Vx: liftoff when Vx >= V_liftoff
            problem.zf[2] = segment.altitude                  # y: climb target
            problem.zf[3] = None                              # Vy: no maximum 
            problem.zf[4] = None                              # m: no maximum
            problem.tf = 0.0                                  # final time

            # check if this problem is well-posed


    elif segment.type.lower() == 'descent':         # Descent segment
        
        options.end_type = "Altitude"
        options.DOFs = 2

    elif segment.type.lower() == 'cruise':          # Cruise segment
        
        options.end_type = segment.end.type
        options.end_value = segment.end.value
        options.DOFs = 2

    elif segment.type.lower() == 'takeoff':         # Takeoff segment (on runway)
        
        problem.tag = "Takeoff"
        problem.f = EquationsOfMotion.Ground1DOF

        # initial conditions
        problem.z0 = ICs

        # final conditions
        problem.zmin = np.zeros(5)
        problem.zmin[0] = None                              # x: no minimum
        problem.zmin[1] = 0.0                               # Vx: must be > 0
        problem.zmin[2] = None                              # y: N/A
        problem.zmin[3] = None                              # Vy: N/A
        problem.zmin[4] = config.Mass_Props.m_flight_min     # m: mimumum mass (out of fuel)
        
        problem.zmax = np.zeros(5)
        problem.zmax[0] = None                              # x: no maximum
        problem.zmax[1] = config.V_liftoff                  # Vx: liftoff when Vx >= V_liftoff
        problem.zmax[2] = None                              # y: N/A
        problem.zmax[3] = None                              # Vy: N/A
        problem.zmax[4] = None                              # m: no maximum

        # check this configuration / segment 
        #y0 = segment.altitude[0]
        #p, T, rho, a, mew = segment.atmosphere.ComputeValues(y0,"all")

        # estimate time step
        problem.h0 = 1.0

    elif segment.type.lower() == 'landing':         # landing segemnt (on runway)
        
        # set up initial conditions
        z0[0] = mission_segment.altitude[0]            # km
        z0[1] = 0.0                                    # km
        options.stop = "Velocity"
        options.EOM = "1DOF"

    elif segment.type.lower() == 'PASS':
        results_segment = EvaluatePASS(config,segment)
    else:
        print "Unknown mission segment type: " + segment.type + "; mission aborted"
    
    if err:
        print "Segment / Configuration check failed"
        results_segment = None
    else:

        print problem.z0
        print problem.h0

        # integrate EOMs:
        solution = RK45(problem,options)

        # package results:
        results_segment = Segment()                     # RESULTS segment container
        results_segment.tag = problem.tag
        results_segment.state.t = solution.t
        results_segment.state.x = solution.z[:,0]
        results_segment.state.Vx = solution.z[:,1]
        results_segment.state.y = solution.z[:,2]
        results_segment.state.Vy = solution.z[:,3]
        results_segment.state.m = solution.z[:,4]
        results_segment.exit = solution.exit
        results_segment.mission_segment = segment
        results_segment.config = config

    return results_segment
