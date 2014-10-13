# test_landing_field_length.py
#
# Created:  Tarik, Carlos, Celso, Jun 2014
# Modified: Emilio Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports
import SUAVE
from SUAVE.Structure            import Data
from SUAVE.Attributes           import Units
from SUAVE.Attributes   import Units
from SUAVE.Methods.Performance.estimate_landing_field_length import estimate_landing_field_length

# package imports
import numpy as np
import pylab as plt

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
    wing.tag = 'Main Wing'

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
    config.wings["Main Wing"].flaps_angle = 15.
    # this configuration is derived from the vehicle.configs.cruise

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    return vehicle
    
def main():

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
        landing_config.mass_properties.landing = weight
        landing_field_length[id_w] = estimate_landing_field_length(vehicle,landing_config,airport)

    truth_LFL = np.array([705.78061286, 766.55136124, 827.32210962, 888.092858, 948.86360638, 
                          1009.63435476, 1070.40510314, 1131.17585152, 1191.9465999, 1252.71734828])
    
    LFL_error = np.max(np.abs(landing_field_length-truth_LFL))
    
    print 'Maximum Landing Field Length Error= %.4e' % LFL_error
    
    title = "LFL vs W"
    plt.figure(1); plt.hold
    plt.plot(w_vec,landing_field_length, 'k-', label = 'Landing Field Length')

    plt.title(title); plt.grid(True)
    legend = plt.legend(loc='lower right', shadow = 'true')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Landing Field Length (m)')
    
    assert( LFL_error   < 1e-5 )

    return 
    
# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    plt.show()
        