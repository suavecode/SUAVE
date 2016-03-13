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
        
    arguments = (segment,state)
    
    #unknowns,infodict,ier,msg = root_finder( iterate,
                            #unknowns,
                            #args = [segment,state],
                            #xtol = state.numerics.tolerance_solution,
                            #full_output=1)    
                            
    unknowns = root_finder( iterate,
                            unknowns,
                            args = [segment,state],
                            xtol = state.numerics.tolerance_solution,
                            fprime=jacobian)      
                            
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
    
    # number of processes
    n = 8
    h = 1e-8
    
    # number of unknowns
    nu = len(unknowns) 
    
    jac = np.zeros((nu,nu))
    baseline = iterate(unknowns, (segment,state))
    
    for ii in xrange(nu):
        unknowns2 = unknowns*1.
        unknowns2[ii] = unknowns[ii]+h
        jac[:,ii] = (iterate(unknowns2, (segment,state)) - baseline)/h

    
    return jac
    
def jacobian(unknowns,(segment,state)):
    
    # number of processes
    n = 8
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
    p.map(e, x)

    # cleanup multiprocessing stuff
    p.close()    
    
    # sort outputs by index
    y = {}
    while not results_queue.empty():
        i,g = results_queue.get()
        y[i] = g
    y = [y[k] for k in sorted(y.keys())]    
    
    jac = (y-base_jac)/h

    return jac
    
        


