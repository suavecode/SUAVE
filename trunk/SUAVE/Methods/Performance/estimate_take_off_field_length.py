# estimate_take_off_field_length.py
#
# Created:  Tarik, Carlos, Celso, Jun 2014
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports
from SUAVE.Core            import Data
from SUAVE.Core import Units

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
                takeoff_constants          - Coefficients for takeoff field lenght equation
                                             [optional. Default values: PASS method]
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
    conditions0 = atmo.compute_values(0.)
    conditions = atmo.compute_values(altitude)
    p = conditions.pressure
    T = conditions.temperature
    rho = conditions.density
    a = conditions.speed_of_sound
    mu = conditions.dynamic_viscosity

    p0 = conditions0.pressure
    T0 = conditions0.temperature
    rho0 = conditions0.density
    a0 = conditions0.speed_of_sound
    mu0 = conditions0.dynamic_viscosity

    T_delta_ISA = T + delta_isa
    sigma_disa = (p/p0) / (T_delta_ISA/T0)
    rho = rho0 * sigma_disa
    a_delta_ISA = atmo.fluid_properties.compute_speed_of_sound(T_delta_ISA)
    mu = 1.78938028e-05 * ((T0 + 120) / T0 ** 1.5) * ((T_delta_ISA ** 1.5) / (T_delta_ISA + 120))
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
        conditions.freestream.viscosity = conditions.dynamic_viscosity
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
    conditions = Data()
    conditions.freestream = Data()
    conditions.propulsion = Data()
    numerics = Data()

    conditions.freestream.dynamic_pressure = np.array([np.atleast_1d(0.5 * rho * speed_for_thrust**2)])
    conditions.freestream.gravity          = np.array([np.atleast_1d(sea_level_gravity)])
    conditions.freestream.velocity         = np.array([np.atleast_1d(speed_for_thrust)])
    conditions.freestream.mach_number      = np.array([np.atleast_1d(speed_for_thrust/ a_delta_ISA)])
    conditions.freestream.temperature      = np.array([np.atleast_1d(T_delta_ISA)])
    conditions.freestream.pressure         = np.array([np.atleast_1d(p)])
    conditions.propulsion.throttle         = np.array([np.atleast_1d(1.)])   

    thrust, mdot, P = analyses.propulsion(conditions,numerics) # total thrust

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
    takeoff_index = V2_speed**2. / (thrust / weight)
    # Calculating takeoff field length
    takeoff_field_length = 0.
    for idx,constant in enumerate(takeoff_constants):
        takeoff_field_length += constant * takeoff_index**idx
        p
    takeoff_field_length = takeoff_field_length * Units.ft
    
    # return
    return takeoff_field_length

# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here

if __name__ == '__main__':

    # ----------------------------------------------------------------------
    #   Imports
    # ----------------------------------------------------------------------
    import SUAVE
    from SUAVE.Core import Units

    # ----------------------------------------------------------------------
    #   Build the Vehicle
    # ----------------------------------------------------------------------
    
     
    def define_vehicle():
    
        # ------------------------------------------------------------------
        #   Initialize the Vehicle
        # ------------------------------------------------------------------
    
        vehicle = SUAVE.Vehicle()
        vehicle.tag = 'EMBRAER E190AR'
    
    
        # ------------------------------------------------------------------
        #   Vehicle-level Properties
        # ------------------------------------------------------------------
    
        # mass properties
        vehicle.mass_properties.takeoff = 50000. #
    
        # basic parameters
        vehicle.reference_area  = 92.    # m^2
    
        # ------------------------------------------------------------------
        #   Main Wing
        # ------------------------------------------------------------------
    
        wing = SUAVE.Components.Wings.Main_Wing()
        wing.tag = 'main_wing'
    
        wing.areas.reference    = vehicle.reference_area
        wing.sweep              = 22. * Units.deg  # deg
        wing.thickness_to_chord = 0.11
        wing.taper              = 0.28          
    
        wing.chords.mean_aerodynamic = 3.66
        wing.areas.affected          = 0.6*wing.areas.reference # part of high lift system
        wing.flap_type   = 'double_slotted'
        wing.flaps_chord  = 0.28
    
        # add to vehicle
        vehicle.append_component(wing)
    
        # ------------------------------------------------------------------
        #  Turbofan
        # ------------------------------------------------------------------
    
        turbofan = SUAVE.Components.Propulsors.TurboFanPASS()
        turbofan.tag = 'TurboFan'
    
        turbofan.propellant = SUAVE.Attributes.Propellants.Jet_A()
        vehicle.fuel_density = turbofan.propellant.density
    
        turbofan.analysis_type                 = '1D'     #
        turbofan.diffuser_pressure_ratio       = 0.98     #
        turbofan.fan_pressure_ratio            = 1.6      #
        turbofan.fan_nozzle_pressure_ratio     = 0.99     #
        turbofan.lpc_pressure_ratio            = 1.9      #
        turbofan.hpc_pressure_ratio            = 10.0     #
        turbofan.burner_pressure_ratio         = 0.95     #
        turbofan.turbine_nozzle_pressure_ratio = 0.99     #
        turbofan.Tt4                           = 1450.0   #
        turbofan.bypass_ratio                  = 5.4      #
        turbofan.thrust.design                 = 25000.0  #
        turbofan.number_of_engines             = 2      #
    
        # turbofan sizing conditions
        sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()
    
        sizing_segment.M   = 0.78          #
        sizing_segment.alt = 10.0         #
        sizing_segment.T   = 218.0        #
        sizing_segment.p   = 0.239*10**5  #
    
        # size the turbofan
        turbofan.engine_sizing_1d(sizing_segment)
    
        # add to vehicle
        vehicle.append_component(turbofan)
    
        # ------------------------------------------------------------------
        #   Simple Propulsion Model
        # ------------------------------------------------------------------
    
        vehicle.propulsion_model = vehicle.propulsors
    
    
        # ------------------------------------------------------------------
        #   Define Configurations
        # ------------------------------------------------------------------
    
        # --- Cruise Configuration ---
        config = vehicle.new_configuration("cruise")
        # this configuration is derived from the baseline vehicle
    
        # --- Takeoff Configuration ---
        config = vehicle.new_configuration("takeoff")
        config.wings['main_wing'].flaps_angle = 15.
        # this configuration is derived from the vehicle.configs.cruise
    
        # ------------------------------------------------------------------
        #   Vehicle Definition Complete
        # ------------------------------------------------------------------
    
        return vehicle
    

    # ----------------------------------------------------------------------
    #   Main
    # ----------------------------------------------------------------------

    # --- Vehicle definition ---
    vehicle = define_vehicle()

    # --- Takeoff Configuration ---
    configuration = vehicle.configs.takeoff
    configuration.wings['main_wing'].flaps.angle =  20. * Units.deg
    configuration.wings['main_wing'].slats.angle  = 25. * Units.deg
    # V2_V2_ratio may be informed by user. If not, use default value (1.2)
    configuration.V2_VS_ratio = 1.21
    # CLmax for a given configuration may be informed by user
    # configuration.maximum_lift_coefficient = 2.XX

    # --- Airport definition ---
    airport = SUAVE.Attributes.Airports.Airport()
    airport.tag = 'airport'
    airport.altitude   =  np.array([0.0]) * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere =  SUAVE.Analyses.Atmospheric.US_Standard_1976()
                          
    w_vec = np.linspace(40000.,52000.,10)
    engines = (2,3,4)
    takeoff_field_length = np.zeros((len(w_vec),len(engines)))
    for id_eng,engine_number in enumerate(engines):
        vehicle.propulsors.TurboFan.number_of_engines = engine_number
        for id_w,weight in enumerate(w_vec):
            configuration.mass_properties.takeoff = weight
            takeoff_field_length[id_w,id_eng] = estimate_take_off_field_length(vehicle,configuration,airport)
            print 'Weight (kg): ',str('%7.0f' % w_vec[id_w]),' ; TOFL (m): ' , str('%6.1f' % takeoff_field_length[id_w,id_eng])

    import pylab as plt
    title = "TOFL vs W"
    plt.figure(1); plt.hold
    plt.plot(w_vec,takeoff_field_length[:,0], 'k-', label = '2 Engines')
    plt.plot(w_vec,takeoff_field_length[:,1], 'r-', label = '3 Engines')
    plt.plot(w_vec,takeoff_field_length[:,2], 'b-', label = '4 Engines')

    plt.title(title); plt.grid(True)
    legend = plt.legend(loc='lower right', shadow = 'true')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Takeoff field length (m)')
    plt.show('True')