## @ingroup Optimization-Package_Setups
# pyopt_setup.py
#
# Created:  Aug 2018, E. Botero
# Modified: Mar 2019, M. Kruger
#           May 2021, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import numpy as np
from SUAVE.Optimization import helper_functions as help_fun
from SUAVE.Core import to_numpy

# ----------------------------------------------------------------------
#  Pyopt_Solve
# ----------------------------------------------------------------------

## @ingroup Optimization-Package_Setups
def Pyoptsparse_Solve(problem,solver='SNOPT',FD='single', sense_step=1.0E-6,  nonderivative_line_search=False):
    """ This converts your SUAVE Nexus problem into a PyOptsparse optimization problem and solves it.
        Pyoptsparse has many algorithms, they can be switched out by using the solver input. 

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        problem                   [nexus()]
        solver                    [str]
        FD (parallel or single)   [str]
        sense_step                [float]
        nonderivative_line_search [bool]

        Outputs:
        outputs                   [list]

        Properties Used:
        None
    """      
   
    # Have the optimizer call the wrapper
    mywrap       = lambda x:PyOpt_Problem(problem,x)
    my_grad_wrap = lambda x,y:PyOpt_Problem_grads(problem,x,y)
   
    inp = to_numpy(problem.optimization_problem.inputs)
    obj = to_numpy(problem.optimization_problem.objective)
    con = to_numpy(problem.optimization_problem.constraints)
   
    if FD == 'parallel':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        myrank = comm.Get_rank()      
   
    # Instantiate the problem and set objective

    try:
        import pyoptsparse as pyOpt
    except:
        raise ImportError('No version of pyOptsparse found')
        
    
    opt_prob = pyOpt.Optimization('SUAVE',mywrap)
    for key in obj.keys():
        opt_prob.addObj(key)    
       
    # Set inputs
    nam  = list(inp.keys())
    inpa = inp.pack_array()
    ini  = inpa[0::5] # Initials
    bndl = inpa[1::5] # Bounds
    bndu = inpa[2::5] # Bounds
    scl  = inpa[3::5] # Scale
        
    # Pull out the constraints and scale them
    bnd_constraints = to_numpy(help_fun.scale_const_bnds(con))
    scaled_constraints = to_numpy(help_fun.scale_const_values(con,bnd_constraints))
    x   = ini/scl
   
    for ii in range(0,len(inp)):
        lbd = (bndl[ii]/scl[ii])
        ubd = (bndu[ii]/scl[ii])
        vartype = 'c'
        opt_prob.addVar(nam[ii],vartype,lower=lbd,upper=ubd,value=x[ii])
       
    # Setup constraints  
    for ii, name in enumerate(con.keys()):
        constraint = con[name]
        edge = scaled_constraints[ii]
       
        if constraint[0]==-1.:
            opt_prob.addCon(name, upper=edge)
        elif constraint[0]==1.:
            opt_prob.addCon(name, lower=edge)
        elif constraint[0]==0.:
            opt_prob.addCon(name, lower=edge,upper=edge)

    # Finalize problem statement and run  
    print(opt_prob)
   
    if solver == 'SNOPT':
        opt = pyOpt.SNOPT()
        CD_step = (sense_step**2.)**(1./3.)  #based on SNOPT Manual Recommendations
        opt.setOption('Function precision', sense_step**2.)
        opt.setOption('Difference interval', sense_step)
        opt.setOption('Central difference interval', CD_step)
        
    elif solver == 'SLSQP':
        opt = pyOpt.SLSQP()
         
    elif solver == 'FSQP':
        opt = pyOpt.FSQP()
        
    elif solver == 'PSQP':
        opt = pyOpt.PSQP()  
        
    elif solver == 'NSGA2':
        opt = pyOpt.NSGA2(pll_type='POA') 
    
    elif solver == 'ALPSO':
        #opt = pyOpt.pyALPSO.ALPSO(pll_type='DPM') #this requires DPM, which is a parallel implementation
        opt = pyOpt.ALPSO()
        
    elif solver == 'CONMIN':
        opt = pyOpt.CONMIN() 
        
    elif solver == 'IPOPT':
        opt = pyOpt.IPOPT()  
    
    elif solver == 'NLPQLP':
        opt = pyOpt.NLQPQLP()     
    
    elif solver == 'NLPY_AUGLAG':
        opt = pyOpt.NLPY_AUGLAG()       
        
    if nonderivative_line_search==True:
        opt.setOption('Nonderivative linesearch')
    if FD == 'parallel':
        outputs = opt(opt_prob, sens='FD',sensMode='pgc')
        
    elif solver == 'SNOPT' or solver == 'SLSQP':
        if problem.use_jax_derivatives:
            outputs = opt(opt_prob, sens=my_grad_wrap)
        else:
            outputs = opt(opt_prob, sens='FD', sensStep = sense_step)
  
    else:
        outputs = opt(opt_prob)        
   
    return outputs


# ----------------------------------------------------------------------
#  Problem Wrapper
# ----------------------------------------------------------------------

## @ingroup Optimization-Package_Setups
def PyOpt_Problem(problem,xdict):
    """ This wrapper runs the SUAVE problem and is called by the PyOpt solver.
        Prints the inputs (x) as well as the objective values and constraints.
        If any values produce NaN then a fail flag is thrown.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        problem   [nexus()]
        x         [array]

        Outputs:
        obj       [float]
        cons      [array]
        fail      [bool]

        Properties Used:
        None
    """      
   
    x = []
    
    for key, val in xdict.items():
        x.append(float(val))
        
    x = np.array(x)

    obj   = problem.objective(x)
    const = problem.all_constraints(x).tolist()
    fail  = np.array(np.isnan(obj.tolist()) or np.isnan(np.array(const).any())).astype(int)
    
    funcs = {}
    
    for ii, obj_name in enumerate(problem.optimization_problem.objective.keys()):
        funcs[obj_name] = obj[ii]
        
    for ii, con_name in enumerate(problem.optimization_problem.constraints.keys()):
        funcs[con_name] = const[ii]
   
    print('Inputs')
    print(x)
    print('Obj')
    print(obj)
    print('Con')
    print(const)
   
    return funcs,fail


## @ingroup Optimization-Package_Setups
def PyOpt_Problem_grads(problem,xdict,ydict):
    """ This wrapper runs the SUAVE problem and is called by the PyOpt solver.
        Prints the inputs (x) as well as the objective values and constraints.
        If any values produce NaN then a fail flag is thrown.

        Assumptions:
        The procedure is jaxable!

        Source:
        N/A

        Inputs:
        problem   [nexus()]
        x         [array]

        Outputs:
        obj       [float]
        cons      [array]
        fail      [bool]

        Properties Used:
        None
    """      
   
    x = list(xdict.values())
    y = list(ydict.values())   # These are the current value of the function
    
    x = np.array(x)
   
    obj   = np.atleast_2d(problem.grad_objective(x).tolist())[0]
    const = np.atleast_2d(problem.jacobian_all_constraints(x).tolist())
    fail  = np.array(np.isnan(obj).any() or np.isnan(np.array(const).any())).astype(int)
    
    # Name of inputs
    inpnam  = list(problem.optimization_problem.inputs.keys()) # Names
    objname = list(problem.optimization_problem.objective.keys())
    conname = list(problem.optimization_problem.constraints.keys())
    
    # Need two for loops. Possibly zips could be used later
    funcs = {}
    
    # Loop over Objectives
    for ii, o_name in enumerate(objname):
        funcs[o_name] = {}
        for jj, i_name in enumerate(inpnam):
            funcs[o_name][i_name] = obj[ii][jj]
    
    # Loop over Objectives
    for ii, c_name in enumerate(conname):
        funcs[c_name] = {}
        for jj, i_name in enumerate(inpnam):
            funcs[c_name][i_name] = obj[ii][jj]


    print('Inputs')
    print(x)
    print('Obj Gradient')
    print(obj)
    print('Constraint Jacobian')
    print(const)
   
    return funcs,fail