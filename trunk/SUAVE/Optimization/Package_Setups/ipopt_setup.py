# ipopt_setup.py
# 
# Created:  Sep 2015, E. Botero 
# Modified: Feb 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import numpy as np

# ----------------------------------------------------------------------
#  Ipopt_Solve
# ----------------------------------------------------------------------

def Ipopt_Solve(problem):
    
    # Pull out the basic problem
    inp = problem.optimization_problem.inputs
    obj = problem.optimization_problem.objective
    con = problem.optimization_problem.constraints
    
    # Number of input variables and constrains
    nvar = len(inp)
    ncon = len(con)
    
    # Set inputs
    ini = inp[:,1] # Initials
    bnd = inp[:,2] # Bounds
    scl = inp[:,3] # Scale
    
    # Scaled initials
    x0 = ini/scl
    x0 = x0.astype(float)
    
    # Nonzero jacobians and hessians, fix this
    nnzj = ncon*nvar
    nnzh = nvar*nvar
     
    # Bounds for inputs and constraints
    flbd = np.zeros_like(ini)
    fubd = np.zeros_like(ini)
    for ii in xrange(0,nvar):
        flbd[ii] = (bnd[ii][0]/scl[ii])
        fubd[ii] = (bnd[ii][1]/scl[ii])

    g_L = np.zeros_like(con)
    g_U = np.zeros_like(con)
    
    # Setup constraints
    for ii in xrange(0,len(con)):
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

def eval_grad_f(x, problem):
    
    grad_f, jac_g = problem.finite_difference(x)
    grad = grad_f.astype(float)

    return grad

def eval_jac_g(x, flag, problem):
    
    if flag:
        matrix = make_structure(problem)
        return matrix
    else:
        grad_f, jac_g = problem.finite_difference(x)
        
        jac_g = np.reshape(jac_g,np.size(jac_g))        
        return jac_g

def eval_f(x, problem):
    
    obj = problem.objective(x)
    obj = obj.astype(float)[0]

    return obj

def eval_g(x, problem):
    
    con = problem.all_constraints(x)
    con = con.astype(float)

    return con

def make_structure(problem):
    
    # Pull out the basic problem
    inp = problem.optimization_problem.inputs
    con = problem.optimization_problem.constraints
    
    # Number of input variables and constrains
    nvar = len(inp)
    ncon = len(con)    
    
    toprow = np.zeros(ncon*nvar).astype(int)
    botrow = np.zeros(ncon*nvar).astype(int)
    
    # All of the rows
    for nn in xrange(0,nvar*ncon):
        val        = (np.floor(nn/nvar)).astype(int)
        toprow[nn] = val
    
    # All of the coulumns
    for nn in xrange(0,nvar*ncon):
        val        = (np.remainder(nn,nvar)).astype(int)
        botrow[nn] = val
    
    array = (toprow.astype(int),botrow.astype(int))
    return array