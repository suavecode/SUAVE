# converge_root.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# Scipy
import scipy
import scipy.optimize
import numpy as np

from SUAVE.Core.Arrays import array_type
from SUAVE.Core.Multi import Evaluator
import multiprocessing as mp
from functools import partial

# ----------------------------------------------------------------------
#  Converge Root
# ----------------------------------------------------------------------

def converge_root(segment,state):
    
    unknowns = state.unknowns.pack_array()
    
    try:
        root_finder = segment.settings.root_finder
    except AttributeError:
        root_finder = scipy.optimize.fsolve 
    
    unknowns,infodict,ier,msg = root_finder( iterate,
                                         unknowns,
                                         args = [segment,state],
                                         xtol = state.numerics.tolerance_solution,
                                         full_output=1)

    if ier!=1:
        print "Segment did not converge. Segment Tag: " + segment.tag
        print "Error Message:\n" + msg
        segment.state.numerics.converged = False
    else:
        segment.state.numerics.converged = True
        
    
    #print 'Calls'
    #print infodict[ 'nfev']
                            
    #unknowns = root_finder( iterate,
                            #unknowns,
                            #args = [segment,state],
                            #xtol = state.numerics.tolerance_solution,
                            #fprime = jacobian2)      
                            
    return
    
# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------
    
def iterate(unknowns,(segment,state)):

    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    else:
        state.unknowns = unknowns
        
    segment.process.iterate(segment,state)
    
    residuals = state.residuals.pack_array()
        
    return residuals 

def jacobian2(unknowns,(segment,state)):
    
    # step size
    h = 1e-8
    
    # number of unknowns
    nu = len(unknowns) 
    
    jac = np.empty((nu,nu))
    baseline = iterate(unknowns, (segment,state))
    
    for ii in xrange(nu):
        unknowns2 = unknowns*1.
        unknowns2[ii] = unknowns[ii]+h
        jac[:,ii] = (iterate(unknowns2, (segment,state)) - baseline)/h
        
    #print jac
    
    #jac = np.array(  [[  1.99993905e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                        #-4.49905601e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                      #[  0.00000000e+00,   1.87597297e+00,  -1.89750438e-03,   8.58102478e-04,
                         #0.00000000e+00,  -4.19321883e+00,   0.00000000e+00,   0.00000000e+00],
                      #[  0.00000000e+00,   4.62506700e-03,   1.62836827e+00,   0.00000000e+00,
                         #0.00000000e+00,   0.00000000e+00,  -3.62796918e+00,   0.00000000e+00],
                      #[  0.00000000e+00,   4.23611146e-03,   2.22203367e-03,   1.51125154e+00,
                         #0.00000000e+00,   0.00000000e+00,   0.00000000e+00,  -3.36759000e+00],
                      #[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                        #-1.21877796e+02,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                      #[  0.00000000e+00,   8.46469561e-03,  -4.60769201e-03,   2.08366657e-03,
                         #0.00000000e+00,  -1.13682446e+02,   0.00000000e+00,   0.00000000e+00],
                      #[  0.00000000e+00,   1.12448717e-02,   3.27702310e-03,   0.00000000e+00,
                         #0.00000000e+00,   0.00000000e+00,  -9.85174388e+01,   0.00000000e+00],
                      #[  0.00000000e+00,   1.03318243e-02,   5.41948708e-03,   1.68487446e-03,
                         #0.00000000e+00,   0.00000000e+00,   0.00000000e+00,  -9.15235235e+01]])
    
    return jac
    
def jacobian(unknowns,(segment,state)):
    
    # number of processes
    n = 4
    h = 1e-8
    
    # number of unknowns
    nu = len(unknowns) 
    
    #jac = np.zeros((nu,nu))
    baseline = iterate(unknowns, (segment,state))
    base_jac = np.tile(baseline, (nu,1))
    
    # Make all of the FD inputs
    inputs = np.tile(unknowns, (nu,1)) + np.eye(nu)*h
    
    # indexify inputs
    x = [ix for ix in enumerate(inputs)]    
    
    # setup multiprocessing stuff
    m = mp.Manager()
    results_queue = m.JoinableQueue()
    p = mp.Pool(n)
    
    args = (segment,state)
    
    # structure to run in parallel
    e = Evaluator(iterate,args,results_queue)   
    
    # run in parallel
    #results_queue2 = p.map_async(e, x)
    p.map(e,x)

    # cleanup multiprocessing stuff
    p.close()    
    
    #sort outputs by index
    y = {}
    while not results_queue.empty():
        i,g = results_queue.get()
        y[i] = g
    
    #results = results_queue2.get()
    #y = np.zeros_like(base_jac)
    #for i in xrange(len(results)):
        #tup   = results[i]
        #y[tup[0],:] = tup[1]
    
    y = [y[k] for k in sorted(y.keys())]    
    
    jac = (y-base_jac)/h

    return jac


from VyPy import parallel as para

def jacobian3(unknowns,(segment,state)):
    
    # number of processes
    n =4
    h = 1e-8
    
    # number of unknowns
    nu = len(unknowns) 
    
    #jac = np.zeros((nu,nu))
    baseline = iterate(unknowns, (segment,state))
    base_jac = np.tile(baseline, (nu,1))
    
    # Make all of the FD inputs
    inputs = np.tile(unknowns, (nu,1)) + np.eye(nu)*h
    
    # indexify inputs
    x = [ix for ix in enumerate(inputs)]    
    
    args = (segment,state)
    
    # structure to run in parallel
    e = Evaluator(iterate,args)   
    
    func = para.MultiTask(e,copies=1)
    y = func(inputs)
    
    jac = (y-base_jac)/h

    return jac


