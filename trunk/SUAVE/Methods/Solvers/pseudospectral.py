""" pseudospectral.py: Pseudospectral collocation-based solver driver for Mission Segments """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from scipy.optimize import root
from SUAVE.Methods.Utilities.Chebyshev import chebyshev_data
from residuals import residuals 
from jacobian_complex import jacobian_complex

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def pseudospectral(problem):

    """  solution = pseudospectral(problem,options): integrate a Segment using Chebyshev pseudospectral method
    
         Inputs:    problem.f = function handle to ODE function of the form z' = f(t,y)     (required)                          (function handle)    
                    problem.FC = function handle to final condition of the form FC(z) = 0   (required)                          (function handle)
                    problem.t0 = initial time                                               (required)                          (float)
                    problem.tf = final time estimate                                        (required)                          (float)
                    problem.z0 = array of initial conditions                                (required)                          (floats)
                    problem.config = vehicle configuration instance                         (required for Mission / Segment)    (class instance)

                    options.tol_solution = solution tolerance                               (required)                          (float)
                    options.tol_BCs = boundary condition tolerance                          (required)                          (float)
                    options.Npoints = number of control points                              (required)                          (int)

         Outputs:   solution.t = time vector                                                                                             (floats)
                    solution.z = m-column array of state variables                                                     (floats)

        """

    # some packing and error checking (needs more work - MC)
    err = False
    if not problem.unpack:
        print "Error: no unpacking function provided. Exiting..."
        err = True; return []

    err = problem.initialize()
    if err:
        print "Error: problem reported with initialization. Exiting..."
        return[]

    x_state_0, x_control_0, dt = problem.unpack(problem.guess)
    
    try: 
        problem.options.N
    except AttributeError:
        print "Warning: number of control points not specified. Using size of initial guess."
        if len(x_state) == 0:
            problem.Nstate = 0
        else:
            if len(np.shape(x_state_0)) == 2:
                problem.options.N, problem.Nstate = np.shape(x_state_0)
            elif len(np.shape(x_state_0)) == 1:
                problem.options.N = np.shape(x_state_0)
                problem.Nstate = 1
    else:
        if len(x_state_0) == 0:
            problem.Nstate = 0
        else:
            if len(np.shape(x_state_0)) == 2:
                rows, problem.Nstate = np.shape(x_state_0)
            elif len(np.shape(x_state_0)) == 1:
                rows = np.shape(x_state_0)
                problem.Nstate = 1
        
                if problem.options.N != rows:
                    print "Warning: number of control points specified does not match size of initial guess. Overriding with size of guess."
                    problem.options.N = rows

    if len(x_control_0) == 0:
        problem.Ncontrol = 0
    else:
        if len(np.shape(x_control_0)) == 2:
            rows, problem.Ncontrol = np.shape(x_control_0)
        elif len(np.shape(x_control_0)) == 1:
            rows = np.shape(x_control_0)

        if problem.options.N != rows:
            print "Warning: number of control points does not match between state and control variables. Exiting..."
            err = True; return []

    if not dt:
        problem.variable_final_time = False
    else:
        problem.variable_final_time = True

    problem.Nvars = problem.Ncontrol + problem.Nstate

    # create "raw" Chebyshev data (0 ---> 1)  
    problem.numerics.t, problem.numerics.D, problem.numerics.I = \
        chebyshev_data(problem.options.N,integration=True)

    # solve system
    solution = root(residuals,
                    problem.guess,
                    args=problem,
                    method="hybr",
                    #jac=jacobian_complex,
                    tol=problem.options.tol_solution)
    
    # confirm final solution
    residuals(solution.x,problem)

    # pack solution
    problem.solution(solution.x)

    return