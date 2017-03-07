# scipy_setup.py
# 
# Created:  Aug 2015, E. Botero 
# Modified: Feb 2017, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import numpy as np
import scipy as sp

# ----------------------------------------------------------------------
#  Something that should become a class at some point
# ----------------------------------------------------------------------

def SciPy_Solve(problem,solver='SLSQP', sense_step = 1.4901161193847656e-08): #1.4901161193847656e-08 is SLSQP default FD step in scipy
    
    inp = problem.optimization_problem.inputs
    obj = problem.optimization_problem.objective
    con = problem.optimization_problem.constraints
    
    # Have the optimizer call the wrapper
    wrapper = lambda x:SciPy_Problem(problem,x)    
    
    # Set inputsq
    nam = inp[:,0] # Names
    ini = inp[:,1] # Initials
    bnd = inp[:,2] # Bounds
    scl = inp[:,3] # Scale
    
    x   = ini/scl
    bnds = np.zeros((len(inp),2))
    for ii in xrange(0,len(inp)):
        # Scaled bounds
        bnds[ii] = (bnd[ii][0]/scl[ii]),(bnd[ii][1]/scl[ii])


    # Finalize problem statement and run
    if solver=='SLSQP':
        outputs = sp.optimize.fmin_slsqp(wrapper,x,f_eqcons=problem.equality_constraint,f_ieqcons=problem.inequality_constraint,bounds=bnds,iter=200, epsilon = sense_step, acc  = sense_step**2)
    else:
        outputs = sp.optimize.minimize(wrapper,x,method=solver)
    
    return outputs


def SciPy_Problem(problem,x):
    
    print 'Inputs'
    print x        
    obj   = problem.objective(x)
    print 'Obj'
    print obj

    
    return obj
