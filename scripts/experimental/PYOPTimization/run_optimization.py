
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Core import (
Data, Container, Data_Exception, Data_Warning,
)

import SUAVE.Plugins.VyPy.optimize as vypy_opt

import pyOpt 
import pyOpt.pySNOPT

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
except:
    raise ImportError('mpi4py is required for parallelization')

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    # setup the interface
    from optimization_interface import setup_interface
    interface = setup_interface()
    
    # setup problem
    problem = setup_problem(interface)
    
    # optimize!
    results = optimize_problem(problem)
    
    return

"""
Results:
variables : 
  projected_span : 30.0
  fuselage_length : 65.0
objectives : 
  fuel_burn : 15693.7348731
equalities : 
inequalities : 
  weight_empty : 62746.4
success : True
messages : 
  exit_flag : 0
  exit_message : Optimization terminated successfully.
  iterations : 7
  run_time : 12.5720000267
"""

# ----------------------------------------------------------------------
#   Setup an Optimization Problem
# ----------------------------------------------------------------------

def setup_problem(interface):
    
    # initialize the problem
    mywrap = lambda inputs:wrap(inputs,vehicle,mission)
    problem = pyOpt.Optimization('B737',mywrap)
    
    # Input design variables with bounds
    problem.addVar('aspect_ratio'  ,'c',value=10.0    ,lower=5.      ,upper=20.0)  
    problem.addVar('reference_area','c',value=125.0   ,lower=70.     ,upper=200.0) 
    problem.addVar('sweep'         ,'c',value=25.0    ,lower=0.      ,upper=60.0)  
    problem.addVar('design_thrust' ,'c',value=24000.0 ,lower=10000.0 ,upper=35000.0)
    problem.addVar('wing_thickness','c',value=0.11    ,lower=0.07    ,upper=0.20)  
    problem.addVar('MTOW'          ,'c',value=79000.0 ,lower=60000.0 ,upper=100000.0)  
    problem.addVar('MZFW_ratio'    ,'c',value=.75     ,lower=0.5     ,upper=1.0)  
    
    # setup objective. Most useless line of code, but its needs to be there...
    problem.addObj('Closed Design')
    
    # setup constraint
    problem.addConGroup('g',5,'i') # Ineguality constraints
    
    #problem.constraints = [
    ##   [ func     , ('tag'                     , '><=', val   ), scl ] ,
        #[ evaluator, ('takeoff_field_length'    , '<'  ,  2180.), 100. ],
        #[ evaluator, ('range_short_field_nmi'   , '>'  ,   650.), 100. ],            
        #[ evaluator, ('range_max_nmi'           , '>'  ,  2725.), 100. ],       
        #[ evaluator, ('max_zero_fuel_margin'    , '>'  ,     0.), 100. ],       
        #[ evaluator, ('available_fuel_margin'   , '>'  ,     0.), 100. ],                       
                   
    #]    
  
    # done!
    return problem
    

# ----------------------------------------------------------------------        
#   Optimize the Problem
# ----------------------------------------------------------------------    

def optimize_problem(problem):
    
    # ------------------------------------------------------------------
    #   Setup Driver
    # ------------------------------------------------------------------    
    
    driver = pyOpt.pySNOPT.SNOPT()
    
    # ------------------------------------------------------------------
    #   Run the Problem
    # ------------------------------------------------------------------        
    fid = open('Results.dat','w')
    
    # This is says we are going to optimize the problem using finite
    # difference gradients in parallel
    results = driver(problem, sens_type='FD',sens_mode='pgc')
    
    print 'Results:'
    print results

    
    # done!
    return results


if __name__ == '__main__':
    main()
