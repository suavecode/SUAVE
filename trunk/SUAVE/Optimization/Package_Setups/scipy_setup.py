## @ingroup Optimization-Package_Setups
# scipy_setup.py
# 
# Created:  Aug 2015, E. Botero 
# Modified: Feb 2017, M. Vegh
#           Mar 2020, E. Botero
#           Jul 2020, M. Clarke
#           May 2021, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import numpy as np
import scipy as sp
from SUAVE.Optimization.Package_Setups.particle_swarm_optimization import particle_swarm_optimization 
from scipy.optimize import NonlinearConstraint
from SUAVE.Optimization import helper_functions as help_fun

# ----------------------------------------------------------------------
#  Something that should become a class at some point
# ----------------------------------------------------------------------

## @ingroup Optimization-Package_Setups
def SciPy_Solve(problem,solver='SLSQP', sense_step = 1.4901161193847656e-08, tolerance = 1e-6, pop_size =  10 , prob_seed = None ):  
    """ This converts your SUAVE Nexus problem into a SciPy optimization problem and solves it
        SciPy has many algorithms, they can be switched out by using the solver input. 

        Assumptions:
        1.4901161193847656e-08 is SLSQP default FD step in scipy

        Source:
        N/A

        Inputs:
        problem                   [nexus()]
        solver                    [str]
        sense_step                [float]

        Outputs:
        outputs                   [list]

        Properties Used:
        None
    """
    
    inp = problem.optimization_problem.inputs
    obj = problem.optimization_problem.objective
    con = problem.optimization_problem.constraints
    
    # Have the optimizer call the wrapper
    wrapper = lambda x:SciPy_Problem(problem,x)    
    
    # Set inputsq
    nam  = inp[:,0] # Names
    ini  = inp[:,1] # Initials
    bndl = inp[:,2] # Bounds
    bndu = inp[:,3] # Bounds
    scl  = inp[:,4] # Scale
    
    x   = ini/scl
    bnds = np.zeros((len(inp),2))
    lb   = np.zeros(len(inp))
    ub   = np.zeros(len(inp))
    de_bnds = []    
    
    for ii in range(0,len(inp)):
        # Scaled bounds
        bnds[ii] = (bndl[ii]/scl[ii]),(bndu[ii]/scl[ii])  
        lb[ii]   = bndl[ii]/scl[ii]
        ub[ii]   = bndu[ii]/scl[ii]
        de_bnds.append((bndl[ii]/scl[ii],bndu[ii]/scl[ii]))  
     
    # Finalize problem statement and run
    if solver=='SLSQP':
        outputs = sp.optimize.fmin_slsqp(wrapper,x,f_eqcons=problem.equality_constraint,f_ieqcons=problem.inequality_constraint,bounds=bnds,\
                                         iter=200, epsilon = sense_step, acc  = tolerance)
    elif solver == 'differential_evolution':
        # Define constraints as a tuple of nonlinear constraints 
        scaled_constraints = []
        aliases            = problem.optimization_problem.aliases
        for ii in range(0,len(con)):
            de_constraint  = con[[ii]] 
            def fun(x):
                problem.evaluate(x)
                constraint_val = help_fun.get_values(problem,de_constraint,aliases)
                return np.atleast_1d(constraint_val)
            
            bound  = help_fun.scale_const_bnds(con)
            if con[ii][1]=='=':
                print('Nonlinear constraints for scipy differential evoultion optimization has '
                      'the general inequality form. Consider rewriting equality constraint as two '
                      'separate inequality constraints')
            
            if con[ii][1]=='>':
                nlc = NonlinearConstraint(fun,bound[ii], np.inf) 
                
            elif con[ii][1]=='<':
                nlc = NonlinearConstraint(fun, -np.inf,bound[ii])
                
            scaled_constraints.append(nlc) 
            
        diff_evo_cons = tuple(scaled_constraints)    
        
        outputs = sp.optimize.differential_evolution(wrapper, bounds= de_bnds, strategy='best1bin', maxiter=1000, popsize = pop_size, \
                                                     tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=prob_seed, callback=None,\
                                                     disp=False, polish=True, init='latinhypercube', atol=0, updating='immediate',\
                                                     workers=1,constraints=diff_evo_cons)
        
    elif solver == 'particle_swarm_optimization':
        outputs = particle_swarm_optimization(wrapper, lb, ub, f_ieqcons=problem.inequality_constraint, kwargs={}, swarmsize=pop_size ,\
                                              omega=0.5, phip=0.5, phig=0.5, maxiter=1000, minstep=1e-4, minfunc=1e-4, debug=False)    
    else:
        outputs = sp.optimize.minimize(wrapper,x,method=solver)
    
    return outputs

## @ingroup Optimization-Package_Setups
def SciPy_Problem(problem,x):
    """ This wrapper runs the SUAVE problem and is called by the Scipy solver.
        Prints the inputs (x) as well as the objective value

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        problem   [nexus()]
        x         [array]

        Outputs:
        obj       [float]

        Properties Used:
        None
    """      
    
    print('Inputs')
    print(x)        
    obj   = problem.objective(x)
    print('Obj')
    print(obj)

    
    return obj

