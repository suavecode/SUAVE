
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import sys
sys.path.append('..') #so you can run it inside a folder
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
    target_range=550*Units.nautical_miles
    # initialize the problem
    problem = vypy_opt.Problem()
    
    # setup variables, list style
    problem.variables = [
    #   [ 'tag'                     ,   x0,               (lb ,            ub    ) , scl      ],
        [ 'aspect_ratio'            , 11.9999999799     , (5.        ,    12.    ) , 'bounds' ], 
        [ 'reference_area'          , 21.             ,          (20.       ,    100.   ) , 'bounds' ],
        [ 'taper'                   , 0.20368           , (.1        ,    .3     ) ,'bounds' ],
        [ 'wing_thickness'          , 0.10916           , (0.07      ,    0.20   ) , 'bounds' ],
        [ 'cruise_range'            , 410.652242348     , (1.        ,    50000. ) , 'bounds'],
        [ 'Vclimb1'                 , 100.96            , (50.       ,    140.   ) , 'bounds'],
        [ 'Vclimb2'                 , 101.732           , (50.       ,    140.   ) , 'bounds'],
        [ 'Vclimb3'                 , 116.751585109     , (50.       ,    144    ) , 'bounds'],
        [ 'cruise_altitude'         , 5.875             , (4.        ,    12.    ) , 'bounds'],
        [ 'climb_alt_fraction_1'    , 0.72387           , (.1        ,    1.     ) , 'bounds'],
        [ 'climb_alt_fraction_2'    , 0.76221           , (.2        ,    1.     ) , 'bounds'],
        [ 'desc_alt_fraction_1'     , 0.3001            , (.1        ,    1.     ) , 'bounds'],
        
        
    ]    

   
   
   
    
    # remember avoids calling the function twice for same inputs
    evaluator = vypy_opt.Remember(interface)
    
    # setup objective
    print evaluator
    problem.objectives = [
    #   [ func     , 'tag'      , scl ],
        [ evaluator, 'GLW', 100. ],
    ]
    
    # setup constraint, list style
    problem.constraints = [
    #   [ func     , ('tag'                     , '><=', val   ), scl ] ,
        [ evaluator, ('takeoff_field_length'    , '<'  ,  1500.)          , 100.   ],           
        [ evaluator, ('landing_field_length'    , '<'  ,  1500.)          , 100.   ],  
        [ evaluator, ('climb_alt_constr'        , '<'  ,   0.  )          , .1     ],
        [ evaluator, ('total_range'             , '>'  ,   target_range  ), 100.  ],
       
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
##    driver = vypy_opt.drivers.BFGS()
##    driver = vypy_opt.drivers.CMA_ES()
##    driver = vypy_opt.drivers.COBYLA()

    driver.verbose = True
    
    
    
    
    # ------------------------------------------------------------------
    #   Run the Problem
    # ------------------------------------------------------------------        
    fid = open('Results.dat','w')
    results = driver.run(problem)
    
    print 'Results:'
    print results

    
    # done!
    return results


if __name__ == '__main__':
    main()
