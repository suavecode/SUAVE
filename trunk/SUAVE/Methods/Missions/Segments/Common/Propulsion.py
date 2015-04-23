
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
#  Update Propulsion
# ----------------------------------------------------------------------

def update_propulsion(segment,state):
    """ update_propulsion()
        updates propulsion conditions

        Inputs -
            segment.analyses.propulsion_model - a callable that will recieve ...
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
    conditions       = state.conditions
    propulsion_model = segment.analyses.propulsion

    # evaluate
    F, mdot, P = propulsion_model(conditions,state.numerics)

    F_vec      =F

    ## unpack results
    #F    = results.thrust_force
    #mdot = atleast_2d_col( results.fuel_mass_rate )
    #P    = atleast_2d_col( results.thurst_power   )

    # pack conditions
    conditions.frames.body.thrust_force_vector[:,:] = F_vec[:,:]
    conditions.propulsion.fuel_mass_rate[:,0]       = mdot[:,0]
    conditions.energies.propulsion_power[:,0]       = P[:,0]

    return
    
