# estimate_landing_field_length.py
#
# Created:  Tarik, Carlos, Celso, Jun 2014
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports
import SUAVE.Methods.Units
from SUAVE.Structure            import Data
from SUAVE.Attributes           import Units

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Compute field length required for landing
# ----------------------------------------------------------------------

def estimate_landing_field_length(vehicle,config,airport):
    """ SUAVE.Methods.Performance.estimate_landing_field_length(vehicle,config,airport):
        Computes the landing field length for a given vehicle condition in a given airport

        Inputs:
            vehicle	 - SUAVE type vehicle

            config   - data dictionary with fields:
                Mass_Properties.landing    - Landing weight to be evaluated
                S                          - Wing Area
                Vref_VS_ratio              - Ratio between Approach Speed and Stall speed
                                             [optional. Default value = 1.23]
                maximum_lift_coefficient   - Maximum lift coefficient for the config
                                             [optional. Calculated if not informed]

    airport   - SUAVE type airport data, with followig fields:
                atmosphere                  - Airport atmosphere (SUAVE type)
                altitude                    - Airport altitude
                delta_isa                   - ISA Temperature deviation


        Outputs:
            landing_field_length            - Landing field length


        Assumptions:
      		- Landing field length calculated according to Torenbeek, E., "Advanced
    Aircraft Design", 2013 (equation 9.25)
            - Considering average aav/g values of two-wheel truck (0.40)
    """

    # ==============================================
        # Unpack
    # ==============================================
    atmo            = airport.atmosphere
    altitude        = airport.altitude * Units.ft
    delta_isa       = airport.delta_isa
    weight          = config.mass_properties.landing
    reference_area  = config.reference_area
    try:
        Vref_VS_ratio = config.Vref_VS_ratio
    except:
        Vref_VS_ratio = 1.23

    # ==============================================
    # Computing atmospheric conditions
    # ==============================================
    p0, T0, rho0, a0, mew0 = atmo.compute_values(0)
    p , T , rho , a , mew  = atmo.compute_values(altitude)
    T_delta_ISA = T + delta_isa
    sigma_disa = (p/p0) / (T_delta_ISA/T0)
    rho = rho0 * sigma_disa
    a_delta_ISA = atmo.fluid_properties.compute_speed_of_sound(T_delta_ISA)
    mew = 1.78938028e-05 * ((T0 + 120) / T0 ** 1.5) * ((T_delta_ISA ** 1.5) / (T_delta_ISA + 120))
    sea_level_gravity = atmo.planet.sea_level_gravity

    # ==============================================
    # Determining vehicle maximum lift coefficient
    # ==============================================
    try:   # aircraft maximum lift informed by user
        maximum_lift_coefficient = config.maximum_lift_coefficient
    except:
        # Using semi-empirical method for maximum lift coefficient calculation
        from SUAVE.Methods.Aerodynamics.Lift.High_lift_correlations import compute_max_lift_coeff

        # Condition to CLmax calculation: 90KTAS @ 10000ft, ISA
        p_stall , T_stall , rho_stall , a_stall , mew_stall  = atmo.compute_values(10000. * Units.ft)
        conditions = Data()
        conditions.rho = rho_stall
        conditions.mew = mew_stall
        conditions.V = 90. * Units.knots
        try:
            maximum_lift_coefficient, induced_drag_high_lift = compute_max_lift_coeff(config,conditions)
            config.maximum_lift_coefficient = maximum_lift_coefficient
        except:
            raise ValueError, "Maximum lift coefficient calculation error. Please, check inputs"

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
        landing_constants[2] =   2.485  / sea_level_gravity  # Two-wheels truck : [ (1.56 / 0.40 + 1.07) / (2*sea_level_gravity) ]
##        landing_constants[2] =   2.9725 / sea_level_gravity  # Four-wheels truck: [ (1.56 / 0.32 + 1.07) / (2*sea_level_gravity) ]

    # Calculating landing field length
    landing_field_length = 0.
    for idx,constant in enumerate(landing_constants):
        landing_field_length += constant * Vref**idx

    # return
    return landing_field_length




# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here

if __name__ == '__main__':

    # ----------------------------------------------------------------------
    #   Imports
    # ----------------------------------------------------------------------
    import SUAVE
    from SUAVE.Attributes   import Units

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
        vehicle.Mass_Properties.takeoff = 50000. #
    
        # basic parameters
        vehicle.delta    = 22.                      # deg
        vehicle.S        = 92.                      # m^2
        vehicle.A_engine = np.pi*( 57*0.0254 /2. )**2.
    
        # ------------------------------------------------------------------
        #   Main Wing
        # ------------------------------------------------------------------
    
        wing = SUAVE.Components.Wings.Main_Wing()
        wing.tag = 'Main Wing'
    
        wing.sref      = vehicle.S     #
        wing.sweep     = vehicle.delta * Units.deg #
        wing.t_c       = 0.11          #
        wing.taper     = 0.28          #
    
    
        wing.chord_mac   = 3.66                  #
        wing.S_affected  = 0.6*wing.sref         # part of high lift system
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
        turbofan.design_thrust                 = 25000.0  #
        turbofan.no_of_engines                 = 2      #
    
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
    
        vehicle.propulsion_model = vehicle.Propulsors
    
    
        # ------------------------------------------------------------------
        #   Define Configurations
        # ------------------------------------------------------------------
    
        # --- Cruise Configuration ---
        config = vehicle.new_configuration("cruise")
        # this configuration is derived from the baseline vehicle
    
        # --- Takeoff Configuration ---
        config = vehicle.new_configuration("takeoff")
        config.wings["Main Wing"].flaps_angle = 15.
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


    # --- Landing Configuration ---
    landing_config = vehicle.configs.takeoff
    landing_config.wings['Main Wing'].flaps_angle =  30. * Units.deg
    landing_config.wings['Main Wing'].slats_angle  = 25. * Units.deg
    # Vref_V2_ratio may be informed by user. If not, use default value (1.23)
    landing_config.Vref_VS_ratio = 1.23
    # CLmax for a given configuration may be informed by user
    # landing_config.maximum_lift_coefficient = 2.XX

    # --- Airport definition ---
    airport = SUAVE.Attributes.Airports.Airport()
    airport.tag = 'airport'
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere =  SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    # =====================================
    # Landing field length evaluation
    # =====================================
    w_vec = np.linspace(20000.,44000.,10)
    landing_field_length = np.zeros_like(w_vec)
    for id_w,weight in enumerate(w_vec):
        landing_config.Mass_Properties.landing = weight
        landing_field_length[id_w] = estimate_landing_field_length(vehicle,landing_config,airport)
        print 'Weight (kg): ',str('%7.0f' % w_vec[id_w]),' ; LFL (m): ' , str('%6.1f' % landing_field_length[id_w])

    import pylab as plt
    title = "LFL vs W"
    plt.figure(1); plt.hold
    plt.plot(w_vec,landing_field_length, 'k-', label = 'Landing Field Length')

    plt.title(title); plt.grid(True)
    legend = plt.legend(loc='lower right', shadow = 'true')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Landing Field Length (m)')
    plt.show('True')

