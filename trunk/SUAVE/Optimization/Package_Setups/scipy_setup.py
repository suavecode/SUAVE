## @ingroup Optimization-Package_Setups
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

## @ingroup Optimization-Package_Setups
def SciPy_Solve(problem,solver='SLSQP', sense_step = 1.4901161193847656e-08, pop_size =  10): #
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
    nam = inp[:,0] # Names
    ini = inp[:,1] # Initials
    bnd = inp[:,2] # Bounds
    scl = inp[:,3] # Scale
    
    x   = ini/scl
    bnds = np.zeros((len(inp),2))
    lb   = np.zeros(len(inp))
    ub   = np.zeros(len(inp))
    de_bnds = []    
    
    for ii in range(0,len(inp)):
        # Scaled bounds
        bnds[ii] = (bnd[ii][0]/scl[ii]),(bnd[ii][1]/scl[ii])  
        lb[ii]   = bnd[ii][0]/scl[ii]
        ub[ii]   = bnd[ii][1]/scl[ii]
        de_bnds.append((bnd[ii][0]/scl[ii],bnd[ii][1]/scl[ii]))      

    # Finalize problem statement and run
    if solver=='SLSQP':
        outputs = sp.optimize.fmin_slsqp(wrapper,x,f_eqcons=problem.equality_constraint,f_ieqcons=problem.inequality_constraint,bounds=bnds,iter=200, epsilon = sense_step, acc  = sense_step**2)
    elif solver == 'differential_evolution':
        outputs = sp.optimize.differential_evolution(wrapper, bounds= de_bnds, strategy='best1bin', maxiter=1000, popsize = pop_size, tol=0.01, mutation=(0.5, 1), recombination=0.9, seed=None, callback=None, disp=False, polish=True, init='latinhypercube', atol=0, updating='immediate', workers=1)
    elif solver == 'particle_swarm_optimization':
        outputs = pso(wrapper, lb, ub, f_ieqcons=problem.inequality_constraint, kwargs={}, swarmsize=pop_size , omega=0.5, phip=0.5, phig=0.5, maxiter=1000, minstep=1e-4, minfunc=1e-4, debug=False)    
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



def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
        minstep=1e-8, minfunc=1e-8, debug=False):
    """
    This function perform a particle swarm optimization (PSO)
   
    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
   
    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal 
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint 
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
   
    """
   
    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
   
    vhigh = np.abs(ub - lb)
    vlow = -vhigh
    
    # Check for constraint function(s) #########################################
    obj = lambda x: func(x, *args, **kwargs)
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = lambda x: np.array([0])
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = lambda x: np.array([y(x, *args, **kwargs) for y in ieqcons])
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = lambda x: np.array(f_ieqcons(x, *args, **kwargs))
        
    def is_feasible(x):
        check = np.all(cons(x)>=0)
        return check
        
    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fp = np.zeros(S)  # best particle function values
    g = []  # best swarm position
    fg = 1e100  # artificial best swarm position starting value
    
    for i in range(S):
        # Initialize the particle's position
        x[i, :] = lb + x[i, :]*(ub - lb)
   
        # Initialize the particle's best known position
        p[i, :] = x[i, :]
       
        # Calculate the objective's value at the current particle's
        fp[i] = obj(p[i, :])
       
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        if i==0:
            g = p[0, :].copy()

        # If the current particle's position is better than the swarm's,
        # update the best swarm position
        if fp[i]<fg and is_feasible(p[i, :]):
            fg = fp[i]
            g = p[i, :].copy()
       
        # Initialize the particle's velocity
        v[i, :] = vlow + np.random.rand(D)*(vhigh - vlow)
       
    # Iterate until termination criterion met ##################################
    it = 1
    while it<=maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))
        for i in range(S):

            # Update the particle's velocity
            v[i, :] = omega*v[i, :] + phip*rp[i, :]*(p[i, :] - x[i, :]) + \
                      phig*rg[i, :]*(g - x[i, :])
                      
            # Update the particle's position, correcting lower and upper bound 
            # violations, then update the objective function value
            x[i, :] = x[i, :] + v[i, :]
            mark1 = x[i, :]<lb
            mark2 = x[i, :]>ub
            x[i, mark1] = lb[mark1]
            x[i, mark2] = ub[mark2]
            fx = obj(x[i, :])
            
            # Compare particle's best position (if constraints are satisfied)
            if fx<fp[i] and is_feasible(x[i, :]):
                p[i, :] = x[i, :].copy()
                fp[i] = fx

                # Compare swarm's best position to current particle's position
                # (Can only get here if constraints are satisfied)
                if fx<fg:
                    if debug:
                        print('New best for swarm at iteration {:}: {:} {:}'.format(it, x[i, :], fx))

                    tmp = x[i, :].copy()
                    stepsize = np.sqrt(np.sum((g-tmp)**2))
                    if np.abs(fg - fx)<=minfunc:
                        print('Stopping search: Swarm best objective change less than {:}'.format(minfunc))
                        return tmp, fx
                    elif stepsize<=minstep:
                        print('Stopping search: Swarm best position change less than {:}'.format(minstep))
                        return tmp, fx
                    else:
                        g = tmp.copy()
                        fg = fx

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
        it += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    
    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    return g, fg



