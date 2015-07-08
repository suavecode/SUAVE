# estimate_take_off_field_length.py
#
# Created:  Tarik, Carlos, Celso, Jun 2014
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports
import SUAVE
from SUAVE.Core            import Data
from SUAVE.Core            import Units

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Compute field length required for takeoff
# ----------------------------------------------------------------------

def estimate_take_off_field_length(vehicle,analyses,airport):
    """ SUAVE.Methods.Performance.estimate_take_off_field_length(vehicle,airport):
        Computes the takeoff field length for a given vehicle condition in a given airport

        Inputs:
            vehicle	 - SUAVE type vehicle

            includes these fields:
                Mass_Properties.takeoff       - Takeoff weight to be evaluated
                S                          - Wing Area
                V2_VS_ratio                - Ratio between V2 and Stall speed
                                             [optional. Default value = 1.20]
                takeoff_constants          - Coefficients for takeoff field length equation
                                             [optional. Default values: AA241 method]
                maximum_lift_coefficient   - Maximum lift coefficient for the config
                                             [optional. Calculated if not informed]

    airport   - SUAVE type airport data, with followig fields:
                atmosphere                  - Airport atmosphere (SUAVE type)
                altitude                    - Airport altitude
                delta_isa                   - ISA Temperature deviation


        Outputs:
            takeoff_field_length            - Takeoff field length


        Assumptions:
            Correlation based.

    """

    # ==============================================
        # Unpack
    # ==============================================
    atmo            = airport.atmosphere
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
    conditions0       = atmo.compute_values(0.)
    atmo_values       = atmo.compute_values(altitude)
    conditions        =SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    
    p   = atmo_values.pressure
    T   = atmo_values.temperature
    rho = atmo_values.density
    a   = atmo_values.speed_of_sound
    mu  =atmo_values.dynamic_viscosity

    p0   = conditions0.pressure
    T0   = conditions0.temperature
    rho0 = conditions0.density
    a0   = conditions0.speed_of_sound
    mu0  = conditions0.dynamic_viscosity

    T_delta_ISA       = T + delta_isa
    sigma_disa        = (p/p0) / (T_delta_ISA/T0)
    rho               = rho0 * sigma_disa
    a_delta_ISA       = atmo.fluid_properties.compute_speed_of_sound(T_delta_ISA)
    mu                = 1.78938028e-05 * ((T0 + 120) / T0 ** 1.5) * ((T_delta_ISA ** 1.5) / (T_delta_ISA + 120))
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
            raise ValueError, "Maximum lift coefficient calculation error. Please, check inputs"

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
        raise ValueError, "No engine found in the vehicle"

    # ==============================================
    # Getting engine thrust
    # ==============================================
    
    state = Data()
    state.conditions = conditions
    state.numerics   = Data()
    conditions.freestream = Data()
    conditions.propulsion = Data()

    conditions.freestream.dynamic_pressure = np.array(np.atleast_1d(0.5 * rho * speed_for_thrust**2))
    conditions.freestream.gravity          = np.array([np.atleast_1d(sea_level_gravity)])
    conditions.freestream.velocity         = np.array(np.atleast_1d(speed_for_thrust))
    conditions.freestream.mach_number      = np.array(np.atleast_1d(speed_for_thrust/ a_delta_ISA))
    conditions.freestream.temperature      = np.array(np.atleast_1d(T_delta_ISA))
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
            print 'The vehicle has more than 4 engines. Using 4 engine correlation. Result may not be correct.'
        else:
            takeoff_constants[0] = 857.4
            takeoff_constants[1] =   2.476
            takeoff_constants[2] =   0.00014
            print 'Incorrect number of engines: {0:.1f}. Using twin engine correlation.'.format(engine_number)

    # Define takeoff index   (V2^2 / (T/W)
    takeoff_index = V2_speed**2. / (thrust[0][0] / weight)
    # Calculating takeoff field length
    takeoff_field_length = 0.
    for idx,constant in enumerate(takeoff_constants):
        takeoff_field_length += constant * takeoff_index**idx
        p
    takeoff_field_length = takeoff_field_length * Units.ft
    # return
    return takeoff_field_length