
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------



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
            thrust_power   - the total propulsion power for each control point

        Assumptions -
            +X out nose
            +Y out starboard wing
            +Z down

    """    
    
    # unpack
    conditions   = state.conditions
    energy_model = segment.analyses.energy

    # evaluate
    F, mdot      = energy_model(conditions,state.numerics)

    #F_vec = state.ones_row(3) * 0.0
    #F_vec[:,0] = F[:,0]

    ## unpack results
    #F    = results.thrust_force
    #mdot = atleast_2d_col( results.fuel_mass_rate )
    #P    = atleast_2d_col( results.thurst_power   )

    # pack conditions
    conditions.frames.body.thrust_force_vector[:,:] = F_vec[:,:]
    conditions.propulsion.fuel_mass_rate[:,0]       = mdot[:,0]
    
    end
    
    