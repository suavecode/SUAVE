""" residuals.py: compute the residuals of a Segment dynamic system """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def residuals(x,problem):

    # preliminaries 
    N = problem.options.N

    # unpack vector
    x_state, x_control, dt = problem.unpack(x)

    # differentiation & integration operators (non-dim t)
    if not problem.variable_final_time:
        dt = problem.dt

    # d/dt and integration[dt] operators
    D = problem.numerics.D/dt
    I = problem.numerics.I*dt

    if problem.Nstate > 0:

        # call user-supplied dynamics function
        rhs = problem.dynamics(x_state,x_control,D,I)

        # evaluate residuals of EOMs
        for j in xrange(problem.Nstate):
            Rs[:,j] = np.dot(D,x_state[:,j]) - rhs[:,j] 
        Rs = Rs[1:].flatten('F')       

    else:
        Rs = []
       
    if problem.Ncontrol > 0:
        
        # call user-supplied constraint functions
        Rc = problem.constraints(x_state,x_control,D,I)
        Rc = Rc.flatten('F')

    else:
        Rc = []

    # append constraints
    R = np.append(Rs,Rc)

    # append final condition if needed
    if problem.variable_final_time: 
         
        Rf = problem.final_condition(x_state,x_control,D,I)
        R = np.append(R,Rf)

    return R
