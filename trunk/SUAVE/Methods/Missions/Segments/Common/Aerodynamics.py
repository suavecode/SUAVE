# Aerodynamics.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

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
                freestream.dynamic_viscosity
                    
    """
    
    # unpack
    conditions            = state.conditions
    h                     = conditions.freestream.altitude
    temperature_deviation = segment.temperature_deviation
    atmosphere            = segment.analyses.atmosphere
    
    # compute
    atmo_data = atmosphere.compute_values(h,temperature_deviation)
    
    # pack
    conditions.freestream.pressure          = atmo_data.pressure
    conditions.freestream.temperature       = atmo_data.temperature
    conditions.freestream.density           = atmo_data.density
    conditions.freestream.speed_of_sound    = atmo_data.speed_of_sound
    conditions.freestream.dynamic_viscosity = atmo_data.dynamic_viscosity
    
    # Gamma
    gamma = segment.analyses.atmosphere.fluid_properties.compute_gamma(atmo_data.temperature,atmo_data.pressure)
    
    # Gas constant
    R = segment.analyses.atmosphere.fluid_properties.gas_specific_constant
    
    # Specific Heat
    Cp = segment.analyses.atmosphere.fluid_properties.compute_cp(atmo_data.temperature,atmo_data.pressure)
    
    # pack
    conditions.freestream.gamma                 = gamma
    conditions.freestream.gas_specific_constant = R
    conditions.freestream.specific_heat         = Cp
    
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
                freestream.dynamic_viscosity

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
    mew  = conditions.freestream.dynamic_viscosity

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
            segment.analyses.aerodynamics_model - a callable that will receive ...
            state.conditions - passed directly to the aerodynamics model

        Outputs -
            lift, drag coefficient, lift drag force, stores to body axis data

        Assumptions -
            +X out nose
            +Y out starboard wing
            +Z down

    """
    
    # unpack
    conditions         = state.conditions
    aerodynamics_model = segment.analyses.aerodynamics
    q                  = state.conditions.freestream.dynamic_pressure
    Sref               = aerodynamics_model.geometry.reference_area
    CLmax              = aerodynamics_model.settings.maximum_lift_coefficient
    
    # call aerodynamics model
    results = aerodynamics_model( state )    
    
    # unpack results
    CL = results.lift.total
    CD = results.drag.total

    CL[q<=0.0] = 0.0
    CD[q<=0.0] = 0.0
    
    # CL limit
    CL[CL>CLmax] = CLmax
    
    CL[CL< -CLmax] = -CLmax
        
    # dimensionalize
    L = state.ones_row(3) * 0.0
    D = state.ones_row(3) * 0.0

    L[:,2] = ( -CL * q * Sref )[:,0]
    D[:,0] = ( -CD * q * Sref )[:,0]

    results.lift_force_vector = L
    results.drag_force_vector = D    

    # pack conditions
    conditions.aerodynamics.lift_coefficient = CL
    conditions.aerodynamics.drag_coefficient = CD
    conditions.frames.wind.lift_force_vector[:,:] = L[:,:] # z-axis
    conditions.frames.wind.drag_force_vector[:,:] = D[:,:] # x-axis


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

