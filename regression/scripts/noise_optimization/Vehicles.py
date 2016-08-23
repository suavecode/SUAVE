# Vehicles.py
# 
# Created:  Nov 2015, Carlos / Tarik
# Modified: Mar 2016, M. Vegh

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
import numpy as np
from SUAVE.Core import Units,Data
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing


# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def setup():
    
    base_vehicle = base_setup()
    configs = configs_setup(base_vehicle)
    
    return configs 
    
    
def base_setup():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Boeing_737800'    


    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff               = 79015.   # kg
    vehicle.mass_properties.takeoff                   = 79015.   # kg
    vehicle.mass_properties.max_zero_fuel             = 75000.
    vehicle.mass_properties.cargo                     = 10000.  * Units.kilogram   
    vehicle.mass_properties.payload                   = 16300.  * Units.kilogram   
    vehicle.mass_properties.max_payload               = 20500. 

    # envelope properties
    vehicle.envelope.ultimate_load = 2.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 124.862       
    vehicle.passengers             = 170
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "medium-range"


    # ------------------------------------------------------------------
    #   Landing gear
    # ------------------------------------------------------------------
    
    #vehicle.landing_gear = Data()
    #vehicle.landing_gear.main_tire_diameter = 1.12000 * Units.m
    #vehicle.landing_gear.nose_tire_diameter = 0.6858 * Units.m
    #vehicle.landing_gear.main_strut_length = 1.8 * Units.m
    #vehicle.landing_gear.nose_strut_length = 1.3 * Units.m
    #vehicle.landing_gear.main_units = 2     #number of main landing gear units
    #vehicle.landing_gear.nose_units = 1     #number of nose landing gear
    #vehicle.landing_gear.main_wheels = 2    #number of wheels on the main landing gear
    #vehicle.landing_gear.nose_wheels = 2    #number of wheels on the nose landing gear  
    
    landing_gear = SUAVE.Components.Landing_Gear.Landing_Gear()
    landing_gear.tag = "main_landing_gear"
    landing_gear.main_tire_diameter = 1.12000 * Units.m
    landing_gear.nose_tire_diameter = 0.6858 * Units.m
    landing_gear.main_strut_length = 1.8 * Units.m
    landing_gear.nose_strut_length = 1.3 * Units.m
    landing_gear.main_units = 2     #number of main landing gear units
    landing_gear.nose_units = 1     #number of nose landing gear
    landing_gear.main_wheels = 2    #number of wheels on the main landing gear
    landing_gear.nose_wheels = 2    #number of wheels on the nose landing gear      
    vehicle.landing_gear=landing_gear
    #vehicle.append_component(landing_gear)
 
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.aspect_ratio            = 10.18
    wing.sweeps.quarter_chord    = 25. * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.16
    wing.span_efficiency         = 0.9

    wing.spans.projected         = 35.66    

    wing.chords.root             = 6.81
    wing.chords.tip              = 1.09
    wing.chords.mean_aerodynamic = 4.235

    wing.areas.reference         = 124.862 

    wing.twists.root             = 4.0 * Units.degrees
    wing.twists.tip              = -4.0 * Units.degrees

    wing.origin                  = [20,0,0]
    wing.aerodynamic_center      = [3,0,0] 

    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True

    wing.dynamic_pressure_ratio  = 1.0

    # ------------------------------------------------------------------
    #   Flaps
    # ------------------------------------------------------------------
    wing.flaps.chord      =  0.30   
    wing.flaps.span_start =  0.10   # ->     wing.flaps.area = 97.1112
    wing.flaps.span_end   =  0.75
    wing.flaps.type       = 'double_sloted'  # -> wing.flaps.number_slots = 2

    # wing sizing
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'

    wing.aspect_ratio            = 6.16
    wing.sweeps.quarter_chord    = 30 * Units.deg
    wing.thickness_to_chord      = 0.08
    wing.taper                   = 0.4
    wing.span_efficiency         = 0.9

    wing.spans.projected         = 14.146

    wing.chords.root             = 3.28
    wing.chords.tip              = 1.31    
    wing.chords.mean_aerodynamic = 8.0

    wing.areas.reference         = 32.488

    wing.twists.root             = 3.0 * Units.degrees
    wing.twists.tip              = 3.0 * Units.degrees  

    wing.origin                  = [50,0,0]
    wing.aerodynamic_center      = [2,0,0]

    wing.vertical                = False 
    wing.symmetric               = True

    wing.dynamic_pressure_ratio  = 0.9  

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'    

    wing.aspect_ratio            = 1.91
    wing.sweeps.quarter_chord    = 25 * Units.deg
    wing.thickness_to_chord      = 0.08
    wing.taper                   = 0.25
    wing.span_efficiency         = 0.9

    wing.spans.projected         = 7.877

    wing.chords.root             = 6.60
    wing.chords.tip              = 1.65
    wing.chords.mean_aerodynamic = 8.0

    wing.areas.reference         = 32.488

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  

    wing.origin                  = [50,0,0]
    wing.aerodynamic_center      = [2,0,0]    

    wing.vertical                = True 
    wing.symmetric               = False
    wing.t_tail                  = False

    wing.dynamic_pressure_ratio  = 1.0

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------

    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    fuselage.number_coach_seats    = vehicle.passengers
    
    fuselage.seats_abreast         = 6
    fuselage.seat_pitch            = 1

    fuselage.fineness.nose         = 1.6
    fuselage.fineness.tail         = 2.

    fuselage.lengths.nose          = 6.4
    fuselage.lengths.tail          = 8.0
    fuselage.lengths.cabin         = 28.85
    fuselage.lengths.total         = 38.02
    fuselage.lengths.fore_space    = 6.
    fuselage.lengths.aft_space     = 5.    

    fuselage.width                 = 3.74

    fuselage.heights.maximum       = 3.74
    fuselage.heights.at_quarter_length          = 3.74
    fuselage.heights.at_three_quarters_length   = 3.74
    fuselage.heights.at_wing_root_quarter_chord = 3.74

    fuselage.areas.side_projected  = 142.1948
    fuselage.areas.wetted          = 446.718
    fuselage.areas.front_projected = 12.57

    fuselage.effective_diameter    = 3.74 

    fuselage.differential_pressure = 5.0e4 * Units.pascal # Maximum differential pressure

    # add to vehicle
    vehicle.append_component(fuselage)


    # ------------------------------------------------------------------
    #   Turbofan Network
    # ------------------------------------------------------------------    

    #instantiate the gas turbine network
    turbofan = SUAVE.Components.Energy.Networks.Turbofan()
    turbofan.tag = 'turbofan'

    # setup
    turbofan.number_of_engines = 2.0
    turbofan.bypass_ratio      = 5.4
    turbofan.engine_length     = 2.71
    turbofan.nacelle_diameter  = 2.05

    #Engine setup for noise module    
    turbofan.core_nozzle_diameter = 0.92
    turbofan.fan_nozzle_diameter  = 1.659
    turbofan.engine_height        = 0.5  #Engine centerline heigh above the ground plane
    turbofan.exa                  = 1    #distance from fan face to fan exit/ fan diameter)
    turbofan.plug_diameter        = 0.1  #dimater of the engine plug 
    turbofan.geometry_xe          = 1. # Geometry information for the installation effects function
    turbofan.geometry_ye          = 1. # Geometry information for the installation effects function   
    turbofan.geometry_Ce          = 2. # Geometry information for the installation effects function
    
    #compute engine areas
    Awet    = 1.1*np.pi*turbofan.nacelle_diameter*turbofan.engine_length 
    
    #assign engine areas
    turbofan.areas.wetted  = Awet
    
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
    compressor.pressure_ratio        = 1.14    

    # add to network
    turbofan.append(compressor)


    # ------------------------------------------------------------------
    #  Component 4 - High Pressure Compressor

    # instantiate
    compressor = SUAVE.Components.Energy.Converters.Compressor()    
    compressor.tag = 'high_pressure_compressor'

    # setup
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio        = 13.415    

    # add to network
    turbofan.append(compressor)


    # ------------------------------------------------------------------
    #  Component 5 - Low Pressure Turbine

    # instantiate
    turbine = SUAVE.Components.Energy.Converters.Turbine()   
    turbine.tag='low_pressure_turbine'

    # setup
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.93     

    # add to network
    turbofan.append(turbine)


    # ------------------------------------------------------------------
    #  Component 6 - High Pressure Turbine

    # instantiate
    turbine = SUAVE.Components.Energy.Converters.Turbine()   
    turbine.tag='high_pressure_turbine'

    # setup
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.93     

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
    combustor.turbine_inlet_temperature = 1450
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
    nozzle.pressure_ratio        = 0.99    

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

    # for noise
    fan.rotation = 1940  #(Need to be calculated based on the thrust level) 

    # add to network
    turbofan.append(fan)


    # ------------------------------------------------------------------
    #Component 10 : thrust (to compute the thrust)
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='compute_thrust'

    #total design thrust (includes all the engines)
    thrust.total_design             = 2*24000. * Units.N #Newtons

    #design sizing conditions
    altitude      = 35000.0*Units.ft
    mach_number   = 0.78 
    isa_deviation = 0.

    # add to network
    turbofan.thrust = thrust

    #size the turbofan
    turbofan_sizing(turbofan,mach_number,altitude)   

    # add  gas turbine network gt_engine to the vehicle 
    vehicle.append_component(turbofan)  
    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    return vehicle

# ----------------------------------------------------------------------
#   Define the Configurations
# ---------------------------------------------------------------------

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
    config.wings['main_wing'].slats.angle = 20. * Units.deg
    config.max_lift_coefficient_factor    = 1. #0.95
    #Noise input for the landing gear
    config.landing_gear.gear_condition    = 'up'       
    config.output_filename                = 'Flyover_' 

    config.propulsors['turbofan'].fan.rotation     = 3470. #N1 speed
    config.propulsors['turbofan'].fan_nozzle.noise_speed  = 315.
    config.propulsors['turbofan'].core_nozzle.noise_speed = 415.

    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Takeoff Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cutback'
    config.wings['main_wing'].flaps.angle = 20. * Units.deg
    config.wings['main_wing'].slats.angle = 20. * Units.deg
    config.max_lift_coefficient_factor    = 1. #0.95
    #Noise input for the landing gear
    config.landing_gear.gear_condition    = 'up'       
    config.output_filename                = 'Cutback_' 

    config.propulsors['turbofan'].fan.rotation     = 2780. #N1 speed
    config.propulsors['turbofan'].fan_nozzle.noise_speed  = 210.
    config.propulsors['turbofan'].core_nozzle.noise_speed = 360.

    configs.append(config)    

    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------

    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'landing'

    config.wings['main_wing'].flaps.angle = 40. * Units.deg
    config.wings['main_wing'].slats.angle = 20. * Units.deg  
    config.max_lift_coefficient_factor    = 1. #0.95
    #Noise input for the landing gear
    config.landing_gear.gear_condition = 'down'    
    config.output_filename             = 'Approach_'

    config.propulsors['turbofan'].fan.rotation     = 2030.  #N1 speed
    config.propulsors['turbofan'].fan_nozzle.noise_speed  = 109.3
    config.propulsors['turbofan'].core_nozzle.noise_speed = 92.

    configs.append(config)

    # ------------------------------------------------------------------
    #   Short Field Takeoff Configuration
    # ------------------------------------------------------------------ 

    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'short_field_takeoff'
    
    config.wings['main_wing'].flaps.angle = 20. * Units.deg
    config.wings['main_wing'].slats.angle = 20. * Units.deg
    config.max_lift_coefficient_factor    = 1. #0.95
  
    configs.append(config)

    # done!
    return configs