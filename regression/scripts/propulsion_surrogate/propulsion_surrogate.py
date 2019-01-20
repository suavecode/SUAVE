# propulsion_surrogate.py
#
# Created:  Jun 2017, E. Botero
# Modified: 

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
    propulsion = Propulsor_Surrogate()
    
    # Build the surrogate
    propulsion.input_file = 'deck.csv'
    propulsion.number_of_engines = 1.
    propulsion.build_surrogate()
    
    # Setup the test point
    state = Data()
    state.conditions = Data()
    state.conditions.freestream = Data()
    state.conditions.propulsion = Data()
    state.conditions.freestream.mach_number = np.array([[0.4]])
    state.conditions.freestream.altitude    = np.array([[2500.]])
    state.conditions.propulsion.throttle    = np.array([[0.75]])
    
    # Evaluate the test point
    results = propulsion.evaluate_thrust(state)
    
    F    = results.thrust_force_vector
    mdot = results.vehicle_mass_rate
    
    # Truth values
    F_truth    = np.array([[ 1223.60069381,     0.        ,    -0.        ]])
    mdot_truth = np.array([[ 717.33608545]])

    # Error check
    error = Data()
    error.Thrust    = np.max(np.abs(F[0,0]-F_truth[0,0]))/F_truth[0,0]
    error.Mass_Rate = np.max(np.abs(mdot[0,0]-mdot_truth[0,0]))/mdot_truth[0,0]
    
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