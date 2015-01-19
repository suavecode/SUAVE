
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
#  Initialize Battery
# ----------------------------------------------------------------------

def initialize_battery(segment,state):
    
    if state.initials:
        energy_initial = state.initials.propulsion.battery_energy[-1,0]
    else:
        #energy_initial = segment.analyses.energy.max_battery_energy
        energy_initial = 0.0
    
    
    state.conditions.propulsion.battery_energy[:,0] = energy_initial

    return
    
    
    