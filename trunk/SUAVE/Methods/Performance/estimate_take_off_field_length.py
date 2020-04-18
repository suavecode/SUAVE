## @ingroup Methods-Performance
# estimate_take_off_field_length.py
#
# Created:  Jun 2014, T. Orra, C. Ilario, Celso, 
# Modified: Apr 2015, M. Vegh 
#           Jan 2016, E. Botero
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports
import SUAVE
from SUAVE.Core            import Data
from SUAVE.Core            import Units

from SUAVE.Analyses.Mission.Segments.Conditions import Aerodynamics,Numerics
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions import windmilling_drag
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions import estimate_2ndseg_lift_drag_ratio
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions import asymmetry_drag

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Compute field length required for takeoff
# ----------------------------------------------------------------------

## @ingroup Methods-Performance
def estimate_take_off_field_length(vehicle,analyses,airport,compute_2nd_seg_climb = 0):
    """ Computes the takeoff field length for a given vehicle configuration in a given airport.
    Also optionally computes the second segment climb gradient.

    Assumptions:
    For second segment climb gradient:
    One engine inoperative
    Only validated for two engine aircraft

    Source:
    http://adg.stanford.edu/aa241/AircraftDesign.html

    Inputs:
    analyses.base.atmosphere               [SUAVE data type]
    airport.
      altitude                             [m]
      delta_isa                            [K]
    vehicle.
      mass_properties.takeoff              [kg]
      reference_area                       [m^2]
      V2_VS_ratio (optional)               [Unitless]
      maximum_lift_coefficient (optional)  [Unitless]
      propulsors.*.number_of_engines       [Unitless]

    Outputs:
    takeoff_field_length                   [m]

    Properties Used:
    N/A
    """        

    # ==============================================
        # Unpack
    # ==============================================
    atmo            = analyses.base.atmosphere
    altitude        = airport.altitude * Units.ft
    delta_isa       = airport.delta_isa
    weight          = vehicle.mass_properties.takeoff
    reference_area  = vehicle.reference_area
    try:
        V2_VS_ratio = vehicle.V2_VS_ratio
    except:
        V2_VS_ratio = 1.20

    # ==============================================
    # Computing atmospheric conditions
    # ==============================================
    atmo_values       = atmo.compute_values(altitude,delta_isa)
    conditions        = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    
    p   = atmo_values.pressure
    T   = atmo_values.temperature
    rho = atmo_values.density
    a   = atmo_values.speed_of_sound
    mu  = atmo_values.dynamic_viscosity
    sea_level_gravity = atmo.planet.sea_level_gravity
    
    # ==============================================
    # Determining vehicle maximum lift coefficient
    # ==============================================
    try:   # aircraft maximum lift informed by user
        maximum_lift_coefficient = vehicle.maximum_lift_coefficient
    except:
        # Using semi-empirical method for maximum lift coefficient calculation
        from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_max_lift_coeff

        # Condition to CLmax calculation: 90KTAS @ 10000ft, ISA
        conditions  = atmo.compute_values(10000. * Units.ft)
        conditions.freestream=Data()
        conditions.freestream.density   = conditions.density
        conditions.freestream.dynamic_viscosity = conditions.dynamic_viscosity
        conditions.freestream.velocity  = 90. * Units.knots
        try:
            maximum_lift_coefficient, induced_drag_high_lift = compute_max_lift_coeff(vehicle,conditions)
            vehicle.maximum_lift_coefficient = maximum_lift_coefficient
        except:
            raise ValueError("Maximum lift coefficient calculation error. Please, check inputs")

    # ==============================================
    # Computing speeds (Vs, V2, 0.7*V2)
    # ==============================================
    stall_speed = (2 * weight * sea_level_gravity / (rho * reference_area * maximum_lift_coefficient)) ** 0.5
    V2_speed    = V2_VS_ratio * stall_speed
    speed_for_thrust  = 0.70 * V2_speed

    # ==============================================
    # Determining vehicle number of engines
    # ==============================================
    engine_number = 0.
    for propulsor in vehicle.propulsors : # may have than one propulsor
        engine_number += propulsor.number_of_engines
    if engine_number == 0:
        raise ValueError("No engine found in the vehicle")

    # ==============================================
    # Getting engine thrust
    # ==============================================    
    state = Data()
    state.conditions = Aerodynamics() 
    state.numerics   = Numerics()
    conditions = state.conditions

    conditions.freestream.dynamic_pressure = np.array(np.atleast_1d(0.5 * rho * speed_for_thrust**2))
    conditions.freestream.gravity          = np.array([np.atleast_1d(sea_level_gravity)])
    conditions.freestream.velocity         = np.array(np.atleast_1d(speed_for_thrust))
    conditions.freestream.mach_number      = np.array(np.atleast_1d(speed_for_thrust/ a))
    conditions.freestream.speed_of_sound   = np.array(a)
    conditions.freestream.temperature      = np.array(np.atleast_1d(T))
    conditions.freestream.pressure         = np.array(np.atleast_1d(p))
    conditions.propulsion.throttle         = np.array(np.atleast_1d(1.))
    
    results = vehicle.propulsors.evaluate_thrust(state) # total thrust
    
    thrust = results.thrust_force_vector

    # ==============================================
    # Calculate takeoff distance
    # ==============================================

    # Defining takeoff distance equations coefficients
    try:
        takeoff_constants = vehicle.takeoff_constants # user defined
    except:  # default values
        takeoff_constants = np.zeros(3)
        if engine_number == 2:
            takeoff_constants[0] = 857.4
            takeoff_constants[1] =   2.476
            takeoff_constants[2] =   0.00014
        elif engine_number == 3:
            takeoff_constants[0] = 667.9
            takeoff_constants[1] =   2.343
            takeoff_constants[2] =   0.000093
        elif engine_number == 4:
            takeoff_constants[0] = 486.7
            takeoff_constants[1] =   2.282
            takeoff_constants[2] =   0.0000705
        elif engine_number >  4:
            takeoff_constants[0] = 486.7
            takeoff_constants[1] =   2.282
            takeoff_constants[2] =   0.0000705
            print('The vehicle has more than 4 engines. Using 4 engine correlation. Result may not be correct.')
        else:
            takeoff_constants[0] = 857.4
            takeoff_constants[1] =   2.476
            takeoff_constants[2] =   0.00014
            print('Incorrect number of engines: {0:.1f}. Using twin engine correlation.'.format(engine_number))

    # Define takeoff index   (V2^2 / (T/W)
    takeoff_index = V2_speed**2. / (thrust[0][0] / weight)
    # Calculating takeoff field length
    takeoff_field_length = 0.
    for idx,constant in enumerate(takeoff_constants):
        takeoff_field_length += constant * takeoff_index**idx
    takeoff_field_length = takeoff_field_length * Units.ft
    
    # calculating second segment climb gradient, if required by user input
    if compute_2nd_seg_climb:
        # Getting engine thrust at V2 (update only speed related conditions)
        state.conditions.freestream.dynamic_pressure = np.array(np.atleast_1d(0.5 * rho * V2_speed**2))
        state.conditions.freestream.velocity         = np.array(np.atleast_1d(V2_speed))
        state.conditions.freestream.mach_number      = np.array(np.atleast_1d(V2_speed/ a))
        results = vehicle.propulsors['turbofan'].engine_out(state)
        thrust = results.thrust_force_vector[0][0]

        # Compute windmilling drag
        windmilling_drag_coefficient = windmilling_drag(vehicle,state)

        # Compute asymmetry drag   
        asymmetry_drag_coefficient = asymmetry_drag(state, vehicle, windmilling_drag_coefficient)
           
        # Compute l over d ratio for takeoff condition, NO engine failure
        l_over_d = estimate_2ndseg_lift_drag_ratio(vehicle) 
        
        # Compute L over D ratio for takeoff condition, WITH engine failure
        clv2 = maximum_lift_coefficient / (V2_VS_ratio) **2
        cdv2_all_engine = clv2 / l_over_d
        cdv2 = cdv2_all_engine + asymmetry_drag_coefficient + windmilling_drag_coefficient
        l_over_d_v2 = clv2 / cdv2
    
        # Compute 2nd segment climb gradient
        second_seg_climb_gradient = thrust / (weight*sea_level_gravity) - 1. / l_over_d_v2

        return takeoff_field_length, second_seg_climb_gradient

    else:
        # return only takeoff_field_length
        return takeoff_field_length