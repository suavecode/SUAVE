
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np


# ----------------------------------------------------------------------
#  Initialize Battery
# ----------------------------------------------------------------------

def initialize_battery(segment,state):
    
    if state.initials:
        energy_initial = state.initials.conditions.propulsion.battery_energy[-1,0]
    else:
        #energy_initial = segment.analyses.energy.max_battery_energy
        energy_initial = 0.0
    
    
    state.conditions.propulsion.battery_energy[:,0] = energy_initial

    return
def update_thrust(segment,state):
    """ update_energy()
        update energy conditions

        Inputs -
            segment.analyses.energy_network - a callable that will recieve ...
            state.conditions         - passed directly to the propulsion model

        Outputs -
            thrust_force   - a 3-column array with rows of total thrust force vectors
                for each control point, in the body frame
            fuel_mass_rate - the total fuel mass flow rate for each control point
            power  -

        Assumptions -


    """    
    
    # unpack
    energy_model = segment.analyses.energy

    # evaluate
    results   = energy_model.evaluate_thrust(state)

    # pack conditions
    conditions = state.conditions
    conditions.frames.body.thrust_force_vector = results.thrust_force_vector
    conditions.weights.vehicle_mass_rate       = results.vehicle_mass_rate
    
    return
    