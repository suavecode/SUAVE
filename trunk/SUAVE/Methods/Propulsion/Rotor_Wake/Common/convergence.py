## @defgroup Methods-Propulsion-Rotor_Wake-Common
# convergence.py
#
# Created:  May 2022, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import jax.numpy as jnp
from jax import lax

# ----------------------------------------------------------------------
#  Newton Function
# ----------------------------------------------------------------------

def simple_newton(function,jac,intial_x,tol=1e-8,limit=1000.,args=()):
    """
    Performs the inside of the while loop in a newton iteration

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:


    Outputs:


    """             
    
    # Set initials and pack into list
    R              = 1.
    ii             = 1
    Xn             = intial_x.flatten()
    Xnp1           = intial_x.flatten()
    damping_factor = 1.
    Full_vector    = [Xn,Xnp1,R,ii,damping_factor]
    
    cond_fun         = lambda Full_vector:cond(Full_vector,tol,limit,function,jac,*args)
    inner_newton_fun = lambda Full_vector:inner_newton(Full_vector,function,jac,*args)
    
    Full_vector = lax.while_loop(cond_fun, inner_newton_fun, Full_vector)

    # Unpack the final versioon
    Xnp1 = Full_vector[1]
    ii   = Full_vector[3]

    return Xnp1, ii


# ----------------------------------------------------------------------
#   Cond
# ----------------------------------------------------------------------

def cond(Full_vector,tol,limit,function,jac,*args):
    """
    Conditions to terminate the newton solver

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:


    Outputs:


    """      
    
    Full_vector = inner_newton(Full_vector,function,jac,*args)
    R           = Full_vector[2]
    ii          = Full_vector[3]
    
    # The other condition is that there have been too many iterations
    cond1 = R<tol
    cond2 = ii>=limit
    
    full_cond = jnp.logical_not(cond1 | cond2)
    
    return full_cond

# ----------------------------------------------------------------------
#  Conditions to terminate the newton solver
# ----------------------------------------------------------------------

def inner_newton(Full_vector,function,jac,*args):
    """
    Performs the inside of the while loop in a newton iteration

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:


    Outputs:


    """       
    
    # Unpack the full vector
    df = Full_vector[4] # damping factor
    Xn = Full_vector[1] # The newest one!
    ii = Full_vector[3] # number of iterations
    
    # Calculate the functional value and the derivative
    f    = jnp.array(function(Xn,*args)).flatten()
    fp   = jnp.diagonal(jnp.array(jac(Xn,*args))).flatten()
    
    # Update to the new point
    Xnp1 = Xn - df*f/fp

    # Take the residual
    R  = jnp.max(jnp.abs(Xnp1-Xn))
    
    # Update the state
    true_fun  = lambda df: df/2
    false_fun = lambda df: df
    df        = lax.cond((R<1e-4)|(ii>8), true_fun, false_fun, df)

    ii+=1    
    
    # Pack the full vector
    Full_vector = [Xn,Xnp1,R,ii,df]

    
    return Full_vector
