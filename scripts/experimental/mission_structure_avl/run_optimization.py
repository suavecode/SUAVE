
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
    problem = vypy_opt.Problem()
    
    # setup variables, list style
    problem.variables = [
    #   [ 'tag'             ,  x0, (lb , ub) , scl      ],
        [ 'projected_span'  , 40., (30.,45.) , 'bounds' ],
        [ 'fuselage_length' , 65., (40.,70.) , 'bounds' ], 
    ]
    
    # remember avoids calling the function twice for same inputs
    evaluator = vypy_opt.Remember(interface)
    
    # setup objective
    problem.objectives = [
    #   [ func     , 'tag'      , scl ],
        [ evaluator, 'fuel_burn', 100. ],
    ]
    
    # setup constraint, list style
    problem.constraints = [
    #   [ func     , ('tag'         , '><=', val   ), scl ] ,
        [ evaluator, ('weight_empty', '<'  , 62000.), 100. ],
    ]
    
    # done!
    return problem
    

# ----------------------------------------------------------------------        
#   Optimize the Problem
# ----------------------------------------------------------------------    

def optimize_problem(problem):
    
    # ------------------------------------------------------------------
    #   Setup Driver
    # ------------------------------------------------------------------    
    
    driver = vypy_opt.drivers.SLSQP()
    driver.verbose = False
    
    # ------------------------------------------------------------------
    #   Run the Problem
    # ------------------------------------------------------------------        

    results = driver.run(problem)
    
    print 'Results:'
    print results

    
    # done!
    return results


if __name__ == '__main__':
    main()
