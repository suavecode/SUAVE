# propulsion_surrogate.py
#
# Created:  Jun 2017, E. Botero
# Modified: Jan 2020, T. MacDonald

#----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------

import SUAVE
from SUAVE.Components.Energy.Networks.Propulsor_Surrogate import Propulsor_Surrogate
from SUAVE.Core import Data

import numpy as np

#----------------------------------------------------------------------
#   The regression script
# ---------------------------------------------------------------------

def main():
    
    # Instantiate a propulsor
    
    
    # Setup the test point
    state = Data()
    state.conditions = Data()
    state.conditions.freestream = Data()
    state.conditions.propulsion = Data()
    state.conditions.freestream.mach_number = np.array([[0.4],[0.4],[0.4],[0.4],[0.4]])
    state.conditions.freestream.altitude    = np.array([[2500.],[2500.],[2500.],[2500.],[2500.]]) # m
    state.conditions.propulsion.throttle    = np.array([[-0.5],[0.005],[0.75],[0.995],[1.5]])    
    
    # Get linear build results
    propulsion = Propulsor_Surrogate()
    propulsion.input_file = 'deck.csv'
    propulsion.number_of_engines = 1.
    propulsion.surrogate_type = 'linear'
    propulsion.build_surrogate()
    results_linear = propulsion.evaluate_thrust(state)
    
    F_linear = results_linear.thrust_force_vector[:,0]  + 1 
    mdot_linear = results_linear.vehicle_mass_rate[:,0]  + 1 
    
    # Get gaussian build results with surrogate extension
    propulsion = Propulsor_Surrogate()
    propulsion.input_file = 'deck.csv'
    propulsion.number_of_engines = 1.
    propulsion.surrogate_type = 'gaussian'
    propulsion.use_extended_surrogate = True
    propulsion.build_surrogate()
    results_gaussian = propulsion.evaluate_thrust(state)    
    
    F_gaussian = results_gaussian.thrust_force_vector[:,0] + 1 
    mdot_gaussian = results_gaussian.vehicle_mass_rate[:,0] + 1
    
    # Truth values
    F_linear_true      = np.array([0.0,  7965.14481619, 12770.34030835, 14350.57238295, 17607.78543469]) + 1
    mdot_linear_true   = np.array([0.0,   0.1231387392, 0.2151082426, 0.2482609107, 0.3211364166]) + 1
    F_gaussian_true    = np.array([0.0,  9427.23973454, 11508.48532115, 10255.76061835, 10676.15981316]) + 1
    mdot_gaussian_true = np.array([0.0,    0.083640387, 0.1928480953, 0.1626323538, 0.1691090013]) + 1

    # Error check
    error = Data()
    error.thrust_linear = np.max(np.abs((F_linear-F_linear_true)/F_linear))
    error.mdot_linear = np.max(np.abs((mdot_linear-mdot_linear_true)/mdot_linear))
    error.thrust_gaussian = np.max(np.abs((F_gaussian-F_gaussian_true)/F_gaussian))
    error.mdot_gaussian = np.max(np.abs((mdot_gaussian-mdot_gaussian_true)/mdot_gaussian))
    
    print('Errors:')
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)
     
    return



# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()