
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
import pyOpt 
import pyOpt.pySNOPT
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
    mywrap = lambda inputs:wrap(inputs,vehicle,mission)
    problem = pyOpt.Optimization('short_range_transport',mywrap)
    target_range=1300*Units.nautical_miles
    # initialize the problem
    #mywrap = lambda inputs:wrap(inputs,vehicle,mission)
    problem = vypy_opt.Problem()
    
    # setup variables, list style

  
    #   [ 'tag'                     ,   x0,               (lb ,            ub    ) , scl      ],
    problem.addVar('aspect_ratio'         ,'c'  , value= 12.0             , lower=5.        ,  upper=  12.  )
    problem.addVar('reference_area'       ,'c'  , value= 27.0000000001    , lower=20.       ,  upper=  100. )
    problem.addVar('taper'                ,'c'  , value= 0.192064084935   , lower=.1        ,  upper=  .3   )
    problem.addVar('wing_thickness'       ,'c'  , value= 0.0940891095763  , lower=0.07      ,  upper=  0.20 )
    problem.addVar('cruise_range'         ,'c'  , value= 1062.96627871    , lower=1.        ,  upper=  50000.)
    problem.addVar('Vclimb1'              ,'c'  , value= 112.333027578    , lower=50.       ,  upper=  140. )
    problem.addVar('Vclimb2'              ,'c'  , value= 115.5508516339    ,lower=(50.       , upper=   140.)
    problem.addVar('Vclimb3'              ,'c'  , value= 116.990689576    , lower=50.       ,  upper=  144  )
    problem.addVar('cruise_altitude'      ,'c'  , value= 9.98216984292    , lower=4.        ,  upper=  12.  )
    problem.addVar('climb_alt_fraction_1' ,'c'  , value= 0.734510828856   , lower=.1        ,  upper=  1.   )
    problem.addVar('climb_alt_fraction_2' ,'c'  , value= 0.837007282348   , lower=.2        ,  upper=  1.   )
    problem.addVar('desc_alt_fraction_1'  ,'c'  , value= 0.364968738014   , lower=.1        ,  upper=  1.   )
    
    # setup objective. Most useless line of code, but its needs to be there...
    problem.addObj('GLW')
    problem.addConGroup('g',4,'i') 
    '''
aspect_ratio         : 12.0
reference_area       : 20.0000000001
taper                : 0.192064084935
wing_thickness       : 0.0940891095763
cruise_range         : 1062.96627871
Vclimb1              : 112.333027578
Vclimb2              : 97.5508516339
Vclimb3              : 116.990689576
cruise_altitude      : 9.98216984292
climb_alt_fraction_1 : 0.734510828856
climb_alt_fraction_2 : 0.837007282348
desc_alt_fraction_1  : 0.364968738014
GLW : 14292.0083279
tofl : 471.292964876
lfl : 1309.36301593
    '''
    
    
    
    '''
 aspect_ratio         : 11.999914278
 reference_area       : 20.0000000001
 taper                : 0.289438709163
 wing_thickness       : 0.109852863522
 cruise_range         : 993.712103625
 Vclimb1              : 88.4487232104
 Vclimb2              : 95.9905677968
 Vclimb3              : 137.894689078
 cruise_altitude      : 11.872385473
 climb_alt_fraction_1 : 0.411998980463
 climb_alt_fraction_2 : 0.968193875956
 desc_alt_fraction_1  : 0.111365084358
 
total_range : 1300.23789982
GLW : 13547.1218784
tofl : 415.137474455
lfl : 1388.78947072
    '''
    '''
    # remember avoids calling the function twice for same inputs
    evaluator=interface
    #evaluator = vypy_opt.Remember(interface)
    
    # setup objective
    #zprint evaluator
    problem.objectives = [
    #   [ func     , 'tag'      , scl ],
        [ evaluator, 'GLW', 10000. ],
    ]
    
    # setup constraint, list style
    problem.constraints = [
    #   [ func     , ('tag'                     , '><=', val   ), scl ] ,
        [ evaluator, ('takeoff_field_length'    , '<'  ,  1500.)          , 100.   ],           
        [ evaluator, ('landing_field_length'    , '<'  ,  1500.)          , 100.   ],  
        [ evaluator, ('climb_alt_constr'        , '<'  ,   0.  )          , .1     ],
        [ evaluator, ('total_range'             , '>'  ,   target_range  ), 1000.  ],
    ]    
    '''
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
    #driver=pyOpt.pySNOPT.SNOPT()
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
