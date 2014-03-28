""" pseudospectral.py: Pseudospectral collocation-based solver driver for Mission Segments """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# numpy imports
import numpy as np
from scipy.optimize import root

# suave functions
from residuals        import residuals
from jacobian_complex import jacobian_complex
from SUAVE.Methods.Utilities.Chebyshev import chebyshev_data
from SUAVE.Methods.Utilities           import atleast_2d_col

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def pseudospectral(problem):
    """  solution = pseudospectral(problem,options)
         integrate a Segment using Chebyshev pseudospectral method
    """
    
    # initialize probme
    problem.initialize()

    # check initial state
    check_initial_state(problem)

    # create "raw" Chebyshev data, t in range [0,1]
    N = problem.options.N
    t, D, I = chebyshev_data( N )
    problem.numerics.t = t 
    problem.numerics.D = D
    problem.numerics.I = I

    # solve system
    solution = root( residuals        ,
                     problem.guess    ,
                     args   = problem ,
                     method = "hybr"  ,
                     #jac    = jacobian_complex ,
                     tol    = problem.options.tol_solution )
    
    # confirm final solution
    residuals(solution.x,problem)

    # pack solution
    problem.solution(solution.x)

    return


# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------

def check_initial_state(problem):
    
    # get initial guess
    x_state_0, x_control_0, dt = problem.unpack(problem.guess)
    
    # make sure at least 2d column vector
    x_state_0   = atleast_2d_col(x_state_0)
    x_control_0 = atleast_2d_col(x_control_0)
    
    
    # check numbers of states
    if not len(x_state_0):
        problem.Nstate = 0    
    else:
        problem.options.N = x_state_0.shape[0]
        problem.Nstate    = x_state_0.shape[1]

    # check numbers of controls
    if not len(x_control_0):
        problem.Ncontrol = 0    
    else:
        problem.options.N = x_control_0.shape[0]
        problem.Ncontrol  = x_control_0.shape[1]        

    # variable final time?
    if not dt:
        problem.variable_final_time = False
    else:
        problem.variable_final_time = True
    
    # total number of unknowns
    problem.Nvars = problem.Ncontrol + problem.Nstate

    return

