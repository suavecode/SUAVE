# pyopt_setup.py
# 
# Created:  Jul 2015, E. Botero 
# Modified:  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE
from SUAVE.Core import Data
import numpy as np
from SUAVE.Optimization import helper_functions as help_fun
# pyopt imports

# ----------------------------------------------------------------------
#  Solve Setup
# ----------------------------------------------------------------------

def Pyopt_Solve(problem,solver='SNOPT',FD='single'):
    
    # Have the optimizer call the wrapper
    mywrap = lambda x:PyOpt_Problem(problem,x)
    
    inp = problem.optimization_problem.inputs
    obj = problem.optimization_problem.objective
    con = problem.optimization_problem.constraints
    
    if FD == 'parallel':
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        myrank = comm.Get_rank()       
    
    # Instantiate the problem and set objective
    import pyOpt 
    opt_prob = pyOpt.Optimization('SUAVE',mywrap)
    opt_prob.addObj(obj[0,0])    
    
    # Set inputs
    nam = inp[:,0] # Names
    ini = inp[:,1] # Initials
    bnd = inp[:,2] # Bounds
    scl = inp[:,3] # Scale
    typ = inp[:,4] # Type
    
    # Pull out the constraints and scale them
    bnd_constraints = help_fun.scale_const_bnds(con)
    scaled_constraints = help_fun.scale_const_values(con,bnd_constraints)
    x   = ini/scl
    
    for ii in xrange(0,len(inp)):
        lbd = (bnd[ii][0]/scl[ii])
        ubd = (bnd[ii][1]/scl[ii])
        #if typ[ii] == 'continuous':
        vartype = 'c'
        #if typ[ii] == 'integer':
            #vartype = 'i'
        opt_prob.addVar(nam[ii],vartype,lower=lbd,upper=ubd,value=x[ii]) 
        
    # Setup constraints  
    for ii in xrange(0,len(con)):
        name = con[ii][0]
        edge = scaled_constraints[ii]
        
        if con[ii][1]=='<':
            opt_prob.addCon(name, type='i', upper=edge)
        elif con[ii][1]=='>':
            opt_prob.addCon(name, type='i', lower=edge,upper=np.inf)
        elif con[ii][1]=='=':
            opt_prob.addCon(name, type='e', equal=edge)

    # Finalize problem statement and run  
    print opt_prob
    
    if solver == 'SNOPT':
        import pyOpt.pySNOPT
        opt = pyOpt.pySNOPT.SNOPT()
    elif solver == 'SLSQP':
        import pyOpt.pySLSQP
        opt = pyOpt.pySLSQP.SLSQP()
    elif solver == 'KSOPT':
        import pyOpt.pyKSOPT
        opt = pyOpt.pyKSOPT.KSOPT() 
    elif solver == 'ALHSO':
        import pyOpt.pyALHSO
        opt = pyOpt.pyALHSO.ALHSO()   
    elif solver == 'FSQP':
        import pyOpt.pyFSQP
        opt = pyOpt.pyFSQP.FSQP()
    elif solver == 'PSQP':
        import pyOpt.pyPSQP
        opt = pyOpt.pyPSQP.PSQP()     
    elif solver == 'NLPQL':
        import pyOpt.pyNLPQL
        opt = pyOpt.pyNLPQL.NLPQL()        
    
    if FD == 'parallel':
        outputs = opt(opt_prob, sens_type='FD',sens_mode='pgc')
    else:
        outputs = opt(opt_prob, sens_type='FD')        
    
    return outputs


# ----------------------------------------------------------------------
#  Problem Wrapper
# ----------------------------------------------------------------------

def PyOpt_Problem(problem,x):
    
    obj   = problem.objective(x)
    const = problem.all_constraints(x).tolist()
    fail  = np.array(np.isnan(obj.tolist()) or np.isnan(np.array(const).any())).astype(int)

        
    print 'Inputs'
    print x
    print 'Obj'
    print obj
    print 'Con'
    print const
    
    return obj,const,fail
