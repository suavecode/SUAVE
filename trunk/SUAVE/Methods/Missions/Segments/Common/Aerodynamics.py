
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np



# ----------------------------------------------------------------------
#  Update Altitude
# ----------------------------------------------------------------------

def update_altitude(segment,state):
    """ Aerodynamics.update_altitude(segment,state)
        updates freestream altitude from inertial position
        
        Inputs:
            state.conditions:
                frames.inertial.position_vector
        Outputs:
            state.conditions:
                freestream.altitude[
                    
    """    
    altitude = -state.conditions.frames.inertial.position_vector[:,2]
    state.conditions.freestream.altitude[:,0] = altitude
    

# ----------------------------------------------------------------------
#  Update Atmosphere
# ----------------------------------------------------------------------

def update_atmosphere(segment,state):
    """ Aerodynamics.update_atmosphere(segment,state)
        computes conditions of the atmosphere at given altitudes
        
        Inputs:
            state.conditions:
                freestream.altitude
            segment.analyses.atmoshere - an atmospheric model
        Outputs:
            state.conditions:
                freestream.pressure
                freestream.temperature
                freestream.density
                freestream.speed_of_sound
                freestream.viscosity
                    
    """
    
    # unpack
    conditions = state.conditions
    h = conditions.freestream.altitude
    atmosphere = segment.analyses.atmosphere
    
    # compute
    atmo_data = atmosphere.compute_values(h)
    
    # pack
    conditions.freestream.pressure       = atmo_data.pressure
    conditions.freestream.temperature    = atmo_data.temperature
    conditions.freestream.density        = atmo_data.density
    conditions.freestream.speed_of_sound = atmo_data.speed_of_sound
    conditions.freestream.viscosity      = atmo_data.dynamic_viscosity
    
    return
    
    
# ----------------------------------------------------------------------
#  Update Freestream
# ----------------------------------------------------------------------

def update_freestream(segment,state):
    """ compute_freestream(condition)
        computes freestream values

        Inputs:
            state.conditions:
                frames.inertial.velocity_vector
                freestream.density
                freestream.speed_of_sound
                freestream.viscosity

        Outputs:
            state.conditions:
                freestream.dynamic pressure
                freestream.mach number
                freestream.reynolds number - DIMENSIONAL - PER UNIT LENGTH - MUST MULTIPLY BY REFERENCE LENGTH
    """
    
    # unpack
    conditions = state.conditions
    Vvec = conditions.frames.inertial.velocity_vector
    rho  = conditions.freestream.density
    a    = conditions.freestream.speed_of_sound
    mew  = conditions.freestream.viscosity

    # velocity magnitude
    Vmag2 = np.sum( Vvec**2, axis=1)[:,None] # keep 2d column vector
    Vmag  = np.sqrt(Vmag2)

    # dynamic pressure
    q = 0.5 * rho * Vmag2 # Pa

    # Mach number
    M = Vmag / a

    # Reynolds number
    Re = rho * Vmag / mew  # per m

    # pack
    conditions.freestream.velocity         = Vmag
    conditions.freestream.mach_number      = M
    conditions.freestream.reynolds_number  = Re
    conditions.freestream.dynamic_pressure = q

    return


# ----------------------------------------------------------------------
#  Update Aerodynamics
# ----------------------------------------------------------------------

def update_aerodynamics(segment,state):
    """ compute_aerodynamics()
        gets aerodynamics conditions

        Inputs -
            segment.analyses.aerodynamics_model - a callable that will recieve ...
            state.conditions - passed directly to the aerodynamics model

        Outputs -
            lift, drag coefficient, lift drag force, stores to body axis data

        Assumptions -
            +X out nose
            +Y out starboard wing
            +Z down

    """
    
    # unpack
    conditions = state.conditions
    aerodynamics_model = segment.analyses.aerodynamics

    # call aerodynamics model
    results = aerodynamics_model( state )    
    #results = aerodynamics_model( state.conditions )    

    # unpack results
    L = results.lift_force_vector
    D = results.drag_force_vector

    # pack conditions
    conditions.frames.wind.lift_force_vector[:,:] = L[:,:] # z-axis
    conditions.frames.wind.drag_force_vector[:,:] = D[:,:] # x-axis

    return


# ----------------------------------------------------------------------
#  Update Stability
# ----------------------------------------------------------------------

def update_stability(segment,state):

    # unpack
    conditions = state.conditions
    stability_model = segment.analyses.stability
    
    # call aerodynamics model
    if stability_model:
        results = stability_model( state.conditions )        
        conditions.stability.update(results)
    
    return

