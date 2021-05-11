## @ingroup Optimization-Package_Setups
# ipopt_setup.py
# 
# Created:  Sep 2015, E. Botero 
# Modified: Feb 2016, M. Vegh
#           May 2021, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import numpy as np

# ----------------------------------------------------------------------
#  Ipopt_Solve
# ----------------------------------------------------------------------

## @ingroup Optimization-Package_Setups
def Ipopt_Solve(problem):
    """Solves a Nexus optimization problem using ipopt

        Assumptions:
        You can actually install ipopt on your machine

        Source:
        N/A

        Inputs:
        problem    [nexus()]

        Outputs:
        result     [array]

        Properties Used:
        None
    """      
    
    # Pull out the basic problem
    inp = problem.optimization_problem.inputs
    obj = problem.optimization_problem.objective
    con = problem.optimization_problem.constraints
    
    # Number of input variables and constrains
    nvar = len(inp)
    ncon = len(con)
    
    # Set inputs
    ini  = inp[:,1] # Initials
    bndl = inp[:,2] # Bounds
    bndu = inp[:,3] # Bounds
    scl  = inp[:,4] # Scale
    
    # Scaled initials
    x0 = ini/scl
    x0 = x0.astype(float)
    
    # Nonzero jacobians and hessians, fix this
    nnzj = ncon*nvar
    nnzh = nvar*nvar
     
    # Bounds for inputs and constraints
    flbd = np.zeros_like(ini)
    fubd = np.zeros_like(ini)
    for ii in range(0,nvar):
        flbd[ii] = (bndl[ii]/scl[ii])
        fubd[ii] = (bndu[ii]/scl[ii])

    g_L = np.zeros_like(con)
    g_U = np.zeros_like(con)
    
    # Setup constraints
    for ii in range(0,len(con)):
        name = con[ii][0]
        edge = con[ii][2]
        if con[ii][1]=='<':
            g_L[ii] = -np.inf
            g_U[ii] = edge
        elif con[ii][1]=='>':
            g_L[ii] = edge
            g_U[ii] = np.inf
        elif con[ii][1]=='=':
            g_L[ii] = edge
            g_U[ii] = edge

    # Instantiate the problem and set objective
    import pyipopt   #import down here to allow SUAVE to run without the user having Ipopt
    
    flbd = flbd.astype(float)
    fubd = fubd.astype(float)
    g_L  = g_L.astype(float)
    g_U  = g_U.astype(float)
    
    # Create the problem
    nlp = pyipopt.create(nvar, flbd, fubd, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g)

    nlp.str_option('derivative_test_print_all','yes')    
    nlp.str_option('derivative_test','first-order')


    # Solve the problem
    result = nlp.solve(x0,problem)
    nlp.close()
    
    return result


# ----------------------------------------------------------------------
#  Wrap the function and FD
# ----------------------------------------------------------------------

## @ingroup Optimization-Package_Setups
def eval_grad_f(x, problem):
    """ Calculate the gradient of the objective function

        Assumptions:
        You can actually install ipopt on your machine

        Source:
        N/A

        Inputs:
        x          [array]
        problem    [nexus()]

        Outputs:
        grad     [array]

        Properties Used:
        None
    """       
    
    grad_f, jac_g = problem.finite_difference(x)
    grad = grad_f.astype(float)

    return grad

## @ingroup Optimization-Package_Setups
def eval_jac_g(x, flag, problem):
    """ Calculate the jacobian of the constraint function
        If flag is used a structure shape is provided to allow ipopt to size the constraints

        Assumptions:
        You can actually install ipopt on your machine

        Source:
        N/A

        Inputs:
        x          [array]
        flag       [bool]
        problem    [nexus()]

        Outputs:
        jac_g      [array]

        Properties Used:
        None
    """     
    
    if flag:
        matrix = make_structure(problem)
        return matrix
    else:
        grad_f, jac_g = problem.finite_difference(x)
        
        jac_g = np.reshape(jac_g,np.size(jac_g))        
        return jac_g

## @ingroup Optimization-Package_Setups
def eval_f(x, problem):
    """ Find the objective

        Assumptions:
        You can actually install ipopt on your machine

        Source:
        N/A

        Inputs:
        x          [array]
        problem    [nexus()]

        Outputs:
        obj        [float]

        Properties Used:
        None
    """       
    
    obj = problem.objective(x)
    obj = obj.astype(float)[0]

    return obj

## @ingroup Optimization-Package_Setups
def eval_g(x, problem):
    """ Find the constraints

        Assumptions:
        You can actually install ipopt on your machine

        Source:
        N/A

        Inputs:
        x          [array]
        problem    [nexus()]

        Outputs:
        con        [array]

        Properties Used:
        None
    """      
    
    con = problem.all_constraints(x)
    con = con.astype(float)

    return con

## @ingroup Optimization-Package_Setups
def make_structure(problem):
    """ Create an array structure to let ipopt know the size of the problem

        Assumptions:
        You can actually install ipopt on your machine

        Source:
        N/A

        Inputs:
        problem    [nexus()]

        Outputs:
        array      [array]

        Properties Used:
        None
    """       
    
    # Pull out the basic problem
    inp = problem.optimization_problem.inputs
    con = problem.optimization_problem.constraints
    
    # Number of input variables and constrains
    nvar = len(inp)
    ncon = len(con)    
    
    toprow = np.zeros(ncon*nvar).astype(int)
    botrow = np.zeros(ncon*nvar).astype(int)
    
    # All of the rows
    for nn in range(0,nvar*ncon):
        val        = (np.floor(nn/nvar)).astype(int)
        toprow[nn] = val
    
    # All of the columns
    for nn in range(0,nvar*ncon):
        val        = (np.remainder(nn,nvar)).astype(int)
        botrow[nn] = val
    
    array = (toprow.astype(int),botrow.astype(int))
    return array