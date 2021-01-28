## @ingroup Methods-Performance
# estimate_landing_field_length.py
#
# Created:  Jun 2014, T. Orra, C. Ilario, Celso, 
# Modified: Apr 2015, M. Vegh 
#           Jan 2016, E. Botero 
#           Mar 2020, M. Clarke
#           Jul 2020, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from   SUAVE.Core import Data, Units
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_max_lift_coeff

import numpy as np

# ----------------------------------------------------------------------
#  Compute field length required for landing
# ----------------------------------------------------------------------

## @ingroup Methods-Performance
def estimate_landing_field_length(vehicle,analyses,airport):
    """ Computes the landing field length for a given vehicle configuration in a given airport.

    Assumptions:
    See source
    Two wheel trucks (code needed for four wheel trucks also included)

    Source:
    Torenbeek, E., "Advanced Aircraft Design", 2013 (equation 9.25)

    Inputs:
    airport.
      atmosphere                           [SUAVE data type]
      altitude                             [m]
      delta_isa                            [K]
    vehicle.
      mass_properties.landing              [kg]
      reference_area                       [m^2]
      maximum_lift_coefficient (optional)  [Unitless]

    Outputs:
    landing_field_length                   [m]

    Properties Used:
    N/A
    """       
   
    # ==============================================
    # Unpack
    # ==============================================
    atmo            = airport.atmosphere
    altitude        = airport.altitude * Units.ft
    delta_isa       = airport.delta_isa
    weight          = vehicle.mass_properties.landing
    reference_area  = vehicle.reference_area
    try:
        Vref_VS_ratio = vehicle.Vref_VS_ratio
    except:
        Vref_VS_ratio = 1.23
        
    # ==============================================
    # Computing atmospheric conditions
    # ==============================================
    atmo_values     = atmo.compute_values(altitude,delta_isa)
    
    p   = atmo_values.pressure
    T   = atmo_values.temperature
    rho = atmo_values.density
    a   = atmo_values.speed_of_sound
    mu  = atmo_values.dynamic_viscosity
    sea_level_gravity = atmo.planet.sea_level_gravity
   
    # ==============================================
    # Determining vehicle maximum lift coefficient
    # ==============================================
    # Condition to CLmax calculation: 90KTAS @ airport
    state = Data()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    state.conditions.freestream = Data()
    state.conditions.freestream.density           = rho
    state.conditions.freestream.velocity          = 90. * Units.knots
    state.conditions.freestream.dynamic_viscosity = mu
    
    settings = analyses.aerodynamics.settings

    maximum_lift_coefficient, induced_drag_high_lift = compute_max_lift_coeff(state,settings,vehicle)

    # ==============================================
    # Computing speeds (Vs, Vref)
    # ==============================================
    stall_speed  = (2 * weight * sea_level_gravity / (rho * reference_area * maximum_lift_coefficient)) ** 0.5
    Vref         = stall_speed * Vref_VS_ratio
    
    # ========================================================================================
    # Computing landing distance, according to Torenbeek equation
    #     Landing Field Length = k1 + k2 * Vref**2
    # ========================================================================================

    # Defining landing distance equation coefficients
    try:
        landing_constants = config.landing_constants # user defined
    except:  # default values - According to Torenbeek book
        landing_constants = np.zeros(3)
        landing_constants[0] = 250.
        landing_constants[1] =   0.
        landing_constants[2] =  2.485  / sea_level_gravity  # Two-wheels truck : [ (1.56 / 0.40 + 1.07) / (2*sea_level_gravity) ]
        #landing_constants[2] =   2.9725 / sea_level_gravity  # Four-wheels truck: [ (1.56 / 0.32 + 1.07) / (2*sea_level_gravity) ] 
    # Calculating landing field length
    landing_field_length = 0.
    for idx,constant in enumerate(landing_constants):
        landing_field_length += constant * Vref**idx
    
    # return
    return landing_field_length
