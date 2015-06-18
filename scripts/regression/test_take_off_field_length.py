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
from SUAVE.Core import Units
from SUAVE.Core import Units
from SUAVE.Methods.Performance.estimate_take_off_field_length import estimate_take_off_field_length

# package imports
import numpy as np
import pylab as plt

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------
def main():

    # ----------------------------------------------------------------------
    #   Main
    # ----------------------------------------------------------------------

    
    vehicle = vehicle_setup()
    configs=configs_setup(vehicle)
    # --- Takeoff Configuration ---
    configuration = configs.takeoff
    configuration.wings['main_wing'].flaps_angle =  20. * Units.deg
    configuration.wings['main_wing'].slats_angle  = 25. * Units.deg
    # V2_V2_ratio may be informed by user. If not, use default value (1.2)
    configuration.V2_VS_ratio = 1.21
   
    #configs_analyses = analyses_setup(configs)
    analyses=base_analysis(vehicle)
    #analyses = SUAVE.Analyses.Analysis.Container()
    #analyses.configs  = configs_analyses
    
    # CLmax for a given configuration may be informed by user
    # configuration.maximum_lift_coefficient = 2.XX

    # --- Airport definition ---
    airport = SUAVE.Attributes.Airports.Airport()
    airport.tag = 'airport'
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere =  SUAVE.Analyses.Atmospheric.US_Standard_1976()
    
    w_vec = np.linspace(40000.,52000.,10)
    engines = (2,3,4)
    takeoff_field_length = np.zeros((len(w_vec),len(engines)))
    
    for id_eng,engine_number in enumerate(engines):
        
        configuration.propulsors.turbo_fan.number_of_engines = engine_number
        
        for id_w,weight in enumerate(w_vec):
            configuration.mass_properties.takeoff = weight
            takeoff_field_length[id_w,id_eng] = estimate_take_off_field_length(configuration,analyses,airport)
  
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
    plt.figure(2); plt.hold
    plt.plot(w_vec,truth_TOFL[:,0], 'k-', label = '2 Engines')
    plt.plot(w_vec,truth_TOFL[:,1], 'r-', label = '3 Engines')
    plt.plot(w_vec,truth_TOFL[:,2], 'b-', label = '4 Engines')
    legend = plt.legend(loc='lower right', shadow = 'true')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Takeoff field length (m)')    
    
    #currently running again,but need to review results
    #assert( TOFL_error   < 1e-5 )

    return 
    
    
def configs_setup(vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------
    
    configs = SUAVE.Components.Configs.Config.Container()
    
    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)
    
    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------
    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cruise'
    
    configs.append(config)
    
    
    # ------------------------------------------------------------------
    #   Takeoff Configuration
    # ------------------------------------------------------------------
    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'takeoff'
    
    config.wings['main_wing'].flaps.angle = 20. * Units.deg
    config.wings['main_wing'].slats.angle = 25. * Units.deg
    
    config.V2_VS_ratio = 1.21
    config.maximum_lift_coefficient = 2.
    
    configs.append(config)
    return configs   

def vehicle_setup():
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'EMBRAER E190AR'  
    
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff               = 51800.0   # kg
    vehicle.mass_properties.operating_empty           = 29100.0   # kg
    vehicle.mass_properties.takeoff                   = 51800.0   # kg
    ## vehicle.mass_properties.max_zero_fuel             = 0.9 * vehicle.mass_properties.max_takeoff 
    vehicle.mass_properties.cargo                     = 0.  * Units.kilogram   
    
    vehicle.mass_properties.center_of_gravity         = [60 * Units.feet, 0, 0]  # Not correct
    vehicle.mass_properties.moments_of_inertia.tensor = [[10 ** 5, 0, 0],[0, 10 ** 6, 0,],[0,0, 10 ** 7]] # Not Correct
    
    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 92.0       
    vehicle.passengers             = 114
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "medium range"
    
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    
    wing.aspect_ratio            = 8.4
    wing.sweep                   = 22.0 * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.16
    wing.span_efficiency         = 1.0
    
    wing.spans.projected         = 27.8    
    
    wing.chords.root             = 5.7057
    wing.chords.tip              = 0.9129
    wing.chords.mean_aerodynamic = 3.8878
                                   
    wing.areas.reference         = 92.0 
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.wetted
    
    wing.twists.root             = 2.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    
    wing.origin                  = [20,0,0]
    wing.aerodynamic_center      = [3,0,0] 
    
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True
    wing.dynamic_pressure_ratio  = 1.0
    
    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'
    
    wing.aspect_ratio            = 5.5
    wing.sweep                   = 34.5 * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.11
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 11.958     #
                                   
    wing.chords.root             = 3.9175
    wing.chords.tip              = 0.4309 
    wing.chords.mean_aerodynamic = 2.6401

    wing.areas.reference         =  26.0  #
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.wetted
    
    wing.twists.root             = 2.0 * Units.degrees
    wing.twists.tip              = 2.0 * Units.degrees  
    
    wing.origin                  = [50,0,0]
    wing.aerodynamic_center      = [2,0,0]
    
    wing.vertical                = False 
    wing.symmetric               = True
    wing.high_lift               = True
    wing.dynamic_pressure_ratio  = 0.9  
    
    # add to vehicle
    vehicle.append_component(wing)
    

    # ------------------------------------------------------------------
    #   Turbofan Network
    # ------------------------------------------------------------------    
    
    #instantiate the gas turbine network
    turbofan = SUAVE.Components.Energy.Networks.Turbofan()
    turbofan.tag = 'turbo_fan'
    
    # setup
    turbofan.number_of_engines = 2.0
    turbofan.design_thrust     = 20300.0
    turbofan.engine_length     = 3.0
    turbofan.nacelle_diameter  = 1.0
    turbofan.bypass_ratio      = 5.4
    
    # working fluid
    turbofan.working_fluid = SUAVE.Attributes.Gases.Air()
    
    
    # ------------------------------------------------------------------
    #   Component 1 - Ram
    
    # to convert freestream static to stagnation quantities
    
    # instantiate
    ram = SUAVE.Components.Energy.Converters.Ram()
    ram.tag = 'ram'
    
    # add to the network
    turbofan.append(ram)


    # ------------------------------------------------------------------
    #  Component 2 - Inlet Nozzle
    
    # instantiate
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag = 'inlet_nozzle'
    
    # setup
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio        = 0.98
    
    # add to network
    turbofan.append(inlet_nozzle)
    
    
    # ------------------------------------------------------------------
    #  Component 3 - Low Pressure Compressor
    
    # instantiate 
    compressor = SUAVE.Components.Energy.Converters.Compressor()    
    compressor.tag = 'low_pressure_compressor'

    # setup
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio        = 1.9     
    
    # add to network
    turbofan.append(compressor)

    
    # ------------------------------------------------------------------
    #  Component 4 - High Pressure Compressor
    
    # instantiate
    compressor = SUAVE.Components.Energy.Converters.Compressor()    
    compressor.tag = 'high_pressure_compressor'
    
    # setup
    compressor.polytropic_efficiency =  0.91
    compressor.pressure_ratio        =  10.0    
    
    # add to network
    turbofan.append(compressor)


    # ------------------------------------------------------------------
    #  Component 5 - Low Pressure Turbine
    
    # instantiate
    turbine = SUAVE.Components.Energy.Converters.Turbine()   
    turbine.tag='low_pressure_turbine'
    
    # setup
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.99     
    
    # add to network
    turbofan.append(turbine)
    
      
    # ------------------------------------------------------------------
    #  Component 6 - High Pressure Turbine
    
    # instantiate
    turbine = SUAVE.Components.Energy.Converters.Turbine()   
    turbine.tag='high_pressure_turbine'

    # setup
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.99     
    
    # add to network
    turbofan.append(turbine)
      
    
    # ------------------------------------------------------------------
    #  Component 7 - Combustor
    
    # instantiate    
    combustor = SUAVE.Components.Energy.Converters.Combustor()   
    combustor.tag = 'combustor'
    
    # setup
    combustor.efficiency                = 0.99 
    combustor.alphac                    = 1.0     
    combustor.turbine_inlet_temperature = 1500
    combustor.pressure_ratio            = 0.95
    combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()    
    
    # add to network
    turbofan.append(combustor)

    
    # ------------------------------------------------------------------
    #  Component 8 - Core Nozzle
    
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    nozzle.tag = 'core_nozzle'
    
    # setup
    nozzle.polytropic_efficiency = 0.95
    nozzle.pressure_ratio        = 0.99    
    
    # add to network
    turbofan.append(nozzle)


    # ------------------------------------------------------------------
    #  Component 9 - Fan Nozzle
    
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    nozzle.tag = 'fan_nozzle'

    # setup
    nozzle.polytropic_efficiency = 0.95
    nozzle.pressure_ratio        = 0.98    
    
    # add to network
    turbofan.append(nozzle)
    
    
    # ------------------------------------------------------------------
    #  Component 10 - Fan
    
    # instantiate
    fan = SUAVE.Components.Energy.Converters.Fan()   
    fan.tag = 'fan'

    # setup
    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio        = 1.7    
    
    # add to network
    turbofan.append(fan)
    
    
    # ------------------------------------------------------------------
    #  Component 10 - Thrust
    
    # to compute thrust
    
    # instantiate
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='thrust'
    
    # setup
    thrust.bypass_ratio                       = 5.4
    thrust.compressor_nondimensional_massflow = 40.0
    thrust.reference_temperature              = 288.15
    thrust.reference_pressure                 = 1.01325*10**5
    thrust.number_of_engines                  = turbofan.number_of_engines   
    
    # add to network
    turbofan.thrust = thrust
    
    
    # add turbofan to vehicle
    vehicle.propulsors.append(turbofan)
    
    # done!!
    return vehicle    

def base_analysis(vehicle):
    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------     
    analyses = SUAVE.Analyses.Vehicle()
   
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)
    
    # ------------------------------------------------------------------
    #  Energy Analysis
    energy  = SUAVE.Analyses.Energy.Energy()
    energy.network=vehicle.propulsors
    analyses.append(energy)
    
    # done!
    return analyses    

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    plt.show()