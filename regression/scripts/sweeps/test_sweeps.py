# test_sweeps.py
#
# Created:  Oct 2017, M. Vegh

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Optimization import carpet_plot, line_plot

import sys
sys.path.append('../Regional_Jet_Optimization')
from Optimize2 import setup
# the analysis functions

#from Boeing_737 import vehicle_setup, configs_setup


# ----------------------------------------------------------------------
#   Run the whole thing
# ----------------------------------------------------------------------
def main():
    
    problem          = setup()
    outputs_sweep    = linear_sweep(problem)
    truth_obj_sweeps = [ [ 6849.69405276 , 6683.05359866]]
    
    #print outputs_sweep
    max_err_sweeps = (np.max(np.abs(outputs_sweep['objective']-truth_obj_sweeps )))
    
    print 'max_err_sweeps = ', max_err_sweeps
    assert(max_err_sweeps<1e-6)
    outputs_carpet = variable_sweep(problem)
    
    #print outputs_carpet
    truth_obj_carp  =  [[ 6705.53342526 , 6892.86563816],[ 7316.47840395 , 6575.80047318]]
    max_err_carp    = np.max(np.abs(outputs_carpet['objective']-truth_obj_carp)) 
    print ' max_err_carp = ',  max_err_carp
    assert(max_err_carp<1e-6)
    return
        

def linear_sweep(problem):
    number_of_points = 2
    outputs = line_plot(problem, number_of_points, plot_obj = 0, plot_const = 0)
    return outputs
    
def variable_sweep(problem):    
    number_of_points=2
    outputs=carpet_plot(problem, number_of_points,  plot_obj = 0, plot_const = 0)  #run carpet plot, suppressing default plots
    return outputs

if __name__ == '__main__':
    main()