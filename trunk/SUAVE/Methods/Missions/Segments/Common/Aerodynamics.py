## @ingroup Methods-Missions-Segments-Common
# Aerodynamics.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero
#           Jul 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Update Altitude
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def update_altitude(segment,state):
    """ Updates freestream altitude from inertial position
        
        Assumptions:
        N/A
        
        Inputs:
            state.conditions:
                frames.inertial.position_vector [meters]
        Outputs:
            state.conditions:
                freestream.altitude             [meters]
      
        Properties Used:
        N/A
                    
    """    
    altitude = -state.conditions.frames.inertial.position_vector[:,2]
    state.conditions.freestream.altitude[:,0] = altitude
    

# ----------------------------------------------------------------------
#  Update Atmosphere
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def update_atmosphere(segment,state):
    """ Computes conditions of the atmosphere at given altitudes
    
        Assumptions:
        N/A
        
        Inputs:
            state.conditions:
                freestream.altitude    [meters]
            segment.analyses.atmoshere [Function]
            
        Outputs:
            state.conditions:
                freestream.pressure          [pascals]
                freestream.temperature       [kelvin]
                freestream.density           [kilogram/meter^3]
                freestream.speed_of_sound    [meter/second]
                freestream.dynamic_viscosity [pascals-seconds]
                
        Properties Used:
        N/A
                                
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
    
    return
    
    
# ----------------------------------------------------------------------
#  Update Freestream
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def update_freestream(segment,state):
    """ Computes freestream values
        
        Assumptions:
        N/A

        Inputs:
            state.conditions:
                frames.inertial.velocity_vector [meter/second]
                freestream.density              [kilogram/meter^3]
                freestream.speed_of_sound       [meter/second]
                freestream.dynamic_viscosity    [pascals-seconds]

        Outputs:
            state.conditions:
                freestream.dynamic pressure     [pascals]
                freestream.mach number          [Unitless]
                freestream.reynolds number      [1/meter]
                               
        Properties Used:
        N/A
    """
    
    # unpack
    conditions = state.conditions
    Vvec = conditions.frames.inertial.velocity_vector
    rho  = conditions.freestream.density
    a    = conditions.freestream.speed_of_sound
    mu   = conditions.freestream.dynamic_viscosity

    # velocity magnitude
    Vmag2 = np.sum( Vvec**2, axis=1)[:,None] # keep 2d column vector
    Vmag  = np.sqrt(Vmag2)

    # dynamic pressure
    q = 0.5 * rho * Vmag2 # Pa

    # Mach number
    M = Vmag / a

    # Reynolds number
    Re = rho * Vmag / mu  # per m

    # pack
    conditions.freestream.velocity         = Vmag
    conditions.freestream.mach_number      = M
    conditions.freestream.reynolds_number  = Re
    conditions.freestream.dynamic_pressure = q

    return


# ----------------------------------------------------------------------
#  Update Aerodynamics
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def update_aerodynamics(segment,state):
    """ Gets aerodynamics conditions
    
        Assumptions:
        +X out nose
        +Y out starboard wing
        +Z down

        Inputs:
            segment.analyses.aerodynamics_model                  [Function]
            aerodynamics_model.settings.maximum_lift_coefficient [unitless]
            aerodynamics_model.geometry.reference_area           [meter^2]
            state.conditions.freestream.dynamic_pressure         [pascals]

        Outputs:
            conditions.aerodynamics.lift_coefficient [unitless]
            conditions.aerodynamics.drag_coefficient [unitless]
            conditions.frames.wind.lift_force_vector [newtons]
            conditions.frames.wind.drag_force_vector [newtons]

        Properties Used:
        N/A
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

## @ingroup Methods-Missions-Segments-Common
def update_stability(segment,state):
    
    """ Initiates the stability model
    
        Assumptions:
        N/A

        Inputs:
            state.conditions           [Data]
            segment.analyses.stability [function]

        Outputs:
        N/A
    
        Properties Used:
        N/A
    """    

    # unpack
    conditions = state.conditions
    stability_model = segment.analyses.stability
    
    # call aerodynamics model
    if stability_model:
        results = stability_model( state.conditions )        
        conditions.stability.update(results)
    
    return

