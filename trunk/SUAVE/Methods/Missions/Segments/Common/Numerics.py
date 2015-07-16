
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Utilities import atleast_2d_col 


# ----------------------------------------------------------------------
#  Initialize Differentials
# ----------------------------------------------------------------------

def initialize_differentials_dimensionless(segment,state):
    
    # unpack
    numerics = state.numerics
    N                     = numerics.number_control_points
    discretization_method = numerics.discretization_method
    
    # get operators
    x,D,I = discretization_method(N,**numerics)
    x = atleast_2d_col(x)
    
    # pack
    numerics.dimensionless.control_points = x
    numerics.dimensionless.differentiate  = D
    numerics.dimensionless.integrate      = I    
    
    return
    

# ----------------------------------------------------------------------
#  Update Differentials
# ----------------------------------------------------------------------

def update_differentials_time(segment,state):
    
    # unpack
    numerics = state.numerics
    x = numerics.dimensionless.control_points
    D = numerics.dimensionless.differentiate
    I = numerics.dimensionless.integrate
    
    # rescale time
    time = state.conditions.frames.inertial.time
    T = time[-1] - time[0]
    t = x * T
    
    # rescale operators
    D = D / T
    I = I * T
    
    # pack
    numerics.time.control_points = t
    numerics.time.differentiate  = D
    numerics.time.integrate      = I

    return
    