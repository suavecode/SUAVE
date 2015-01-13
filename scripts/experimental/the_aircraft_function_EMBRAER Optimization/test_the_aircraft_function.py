
# test_the_aircraft_function.py
#
# Created:  Carlos, Tarik, Sept 2014
# Modified:

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Attributes import Units
from SUAVE.Core import Data

import numpy as np

from full_setup                     import full_setup
from the_aircraft_function_EMBRAER  import the_aircraft_function_EMBRAER
from post_process                   import post_process

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
     #inputs = area, sweep, thickness, thrust
    inputs = np.array([92.00,22.0 * Units.deg, 0.11, 20300.0])
    
    # Have the optimizer call the wrapper
    mywrap = lambda inputs:wrap(inputs)
    
    opt_prob = pyOpt.Optimization('Embraer',mywrap)
    opt_prob.addObj('Fuel')    
    opt_prob.addVar('Wing Area','c',lower=70,upper=110,value=inputs[0])
    opt_prob.addVar('Wing Sweep','c',lower=15,upper=35,value=inputs[1])
    opt_prob.addVar('Wing thickness','c',lower=0.10,upper=0.15,value=inputs[2])
    opt_prob.addVar('Engine Thrust','c',lower=18000,upper=25000,value=inputs[3])
    opt_prob.addConGroup('g',4,'i')
    
    opt = pyOpt.pySLSQP.SLSQP()  


    post_process(vehicle,mission,results)
    
    outputs = opt(opt_prob, sens_type='FD',sens_mode='pgc')
    
    print outputs

    return

def wrap(inputs):
    
    print inputs
    
    # Remake the vehicle
    vehicle, mission = full_setup(inputs)
    
    # Run the vehicle
    results = the_aircraft_function_EMBRAER(vehicle,mission)
    
    #
    TO = results.field_length.takeoff
    Land = results.field_length.landing
    results.design_payload_mission
    results.short_field
    results.mission_for_fuel    
    
    objective = results.mission_for_fuel.fuel
    
    constraints = [0,0,0,0]
    
    constraints[0] = TO - 2209.36308575
    
    fail = 0
    
    print('Objective')
    print objective
    print('Constraints')
    print constraints
    
    
    return objective,constraints,fail
    


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()
##    plt.show(block=True) # here so as to not block the regression test