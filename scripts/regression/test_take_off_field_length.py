# test_take_off_field_length.py
#
# Created:  Tarik, Carlos, Celso, Jun 2014
# Modified: Emilio Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports
import SUAVE
from SUAVE.Core            import Data
from SUAVE.Attributes           import Units
from SUAVE.Attributes   import Units
from SUAVE.Methods.Performance.estimate_take_off_field_length import estimate_take_off_field_length

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
    
def main():

    # ----------------------------------------------------------------------
    #   Main
    # ----------------------------------------------------------------------

    # --- Vehicle definition ---
    vehicle = define_vehicle()

    # --- Takeoff Configuration ---
    configuration = vehicle.configs.takeoff
    configuration.wings['main_wing'].flaps_angle =  20. * Units.deg
    configuration.wings['main_wing'].slats_angle  = 25. * Units.deg
    # V2_V2_ratio may be informed by user. If not, use default value (1.2)
    configuration.V2_VS_ratio = 1.21
    # CLmax for a given configuration may be informed by user
    # configuration.maximum_lift_coefficient = 2.XX

    # --- Airport definition ---
    airport = SUAVE.Attributes.Airports.Airport()
    airport.tag = 'airport'
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere =  SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    w_vec = np.linspace(40000.,52000.,10)
    engines = (2,3,4)
    takeoff_field_length = np.zeros((len(w_vec),len(engines)))
    for id_eng,engine_number in enumerate(engines):
        vehicle.propulsors.TurboFan.number_of_engines = engine_number
        for id_w,weight in enumerate(w_vec):
            configuration.mass_properties.takeoff = weight
            takeoff_field_length[id_w,id_eng] = estimate_take_off_field_length(vehicle,configuration,airport)

    truth_TOFL = np.array([[  850.19992906,   567.03906016,   411.69975426],
                           [  893.11528215,   592.95224563,   430.3183183 ],
                           [  937.78501446,   619.84892797,   449.62233042],
                           [  984.23928481,   647.73943813,   469.61701137],
                           [ 1032.50914159,   676.63434352,   490.30766781],
                           [ 1082.62653274,   706.54445208,   511.69969462],
                           [ 1134.62431541,   737.48081617,   533.79857721],
                           [ 1188.53626527,   769.45473631,   556.60989361],
                           [ 1244.39708554,   802.47776475,   580.13931657],
                           [ 1302.24241572,   836.56170886,   604.39261547]])
    
    TOFL_error = np.max(np.abs(truth_TOFL-takeoff_field_length))
    
    print 'Maximum Take OFF Field Length Error= %.4e' % TOFL_error
    
    
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
    
    assert( TOFL_error   < 1e-5 )

    return 
    
# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    plt.show()