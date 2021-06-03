# tut_payload_range.py
#
# Created:  Aug 2014, SUAVE Team
# Modified: Apr 2016, T. Orra
#           Aug 2017, E. Botero

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data 
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Methods.Performance  import payload_range
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform

import numpy as np
import pylab as plt

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():

    # define the problem
    configs, analyses = full_setup()

    configs.finalize()
    analyses.finalize()

    # weight analysis
    weights = analyses.configs.base.weights
    breakdown = weights.evaluate()

    # mission analysis
    mission = analyses.missions
    results = mission.evaluate()

    # run payload diagram
    config = configs.base
    cruise_segment_tag = "cruise"
    reserves = 1750.
    payload_range_results = payload_range(config,mission,cruise_segment_tag,reserves)

    # plot the results
    plot_mission(results)

    return

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    # vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = mission_setup(configs_analyses)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = mission

    return configs, analyses

# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):

    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis

    # adjust analyses for configs

    # takeoff_analysis
    analyses.takeoff.aerodynamics.drag_coefficient_increment = 0.1000

    return analyses

def base_analysis(vehicle):

    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------
    analyses = SUAVE.Analyses.Vehicle()

    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    # ------------------------------------------------------------------
    #  Weights
    weights = SUAVE.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.000
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Stability Analysis
    stability = SUAVE.Analyses.Stability.Fidelity_Zero()
    stability.geometry = vehicle
    analyses.append(stability)

    # ------------------------------------------------------------------
    #  Propulsion Analysis
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.propulsors
    analyses.append(energy)

    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)

    return analyses

# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------
def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Embraer_E190AR'

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------

    # mass properties (http://www.embraercommercialaviation.com/AircraftPDF/E190_Weights.pdf)
    vehicle.mass_properties.max_takeoff               = 51800.   # kg
    vehicle.mass_properties.operating_empty           = 27837.   # kg
    vehicle.mass_properties.takeoff                   = 51800.   # kg
    vehicle.mass_properties.max_zero_fuel             = 40900.   # kg
    vehicle.mass_properties.max_payload               = 13063.   # kg
    vehicle.mass_properties.max_fuel                  = 12971.   # kg
    vehicle.mass_properties.cargo                     =     0.0  # kg

    vehicle.mass_properties.center_of_gravity         = [[16.8, 0, 1.6]]
    vehicle.mass_properties.moments_of_inertia.tensor = [[10 ** 5, 0, 0],[0, 10 ** 6, 0,],[0,0, 10 ** 7]] 

    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 92.
    vehicle.passengers             = 106
    vehicle.systems.control        = "fully powered"
    vehicle.systems.accessories    = "medium range"


    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------
    wing                         = SUAVE.Components.Wings.Main_Wing()
    wing.tag                     = 'main_wing'
    wing.areas.reference         = 92.0
    wing.aspect_ratio            = 8.4
    wing.chords.root             = 6.2
    wing.chords.tip              = 1.44
    wing.sweeps.quarter_chord    = 23.0 * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.28
    wing.dihedral                = 5.00 * Units.deg
    wing.spans.projected         = 28.72
    wing.origin                  = [[13.0,0,-1.50]]
    wing.vertical                = False
    wing.symmetric               = True       
    wing.high_lift               = True
    wing.areas.exposed           = 0.80 * wing.areas.wetted        
    wing.twists.root             = 2.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees    
    wing.dynamic_pressure_ratio  = 1.0
    
    # control surfaces -------------------------------------------
    flap                       = SUAVE.Components.Wings.Control_Surfaces.Flap() 
    flap.tag                   = 'flap' 
    flap.span_fraction_start   = 0.11
    flap.span_fraction_end     = 0.85
    flap.deflection            = 0.0 * Units.deg 
    flap.chord_fraction        = 0.28    
    flap.configuration_type    = 'double_slotted'
    wing.append_control_surface(flap)   
        
    slat                       = SUAVE.Components.Wings.Control_Surfaces.Slat()
    slat.tag                   = 'slat' 
    slat.span_fraction_start   = 0.324 
    slat.span_fraction_end     = 0.963     
    slat.deflection            = 1.0 * Units.deg 
    slat.chord_fraction        = 0.1   
    wing.append_control_surface(slat) 
    
    wing                         = wing_planform(wing)
    
    wing.areas.exposed           = 0.80 * wing.areas.wetted
    wing.twists.root             = 2.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees    
    wing.dynamic_pressure_ratio  = 1.0   

    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Horizontal_Tail()
    wing.tag = 'horizontal_stabilizer'
    wing.areas.reference         = 26.0
    wing.aspect_ratio            = 5.5
    wing.sweeps.quarter_chord    = 34.5 * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.11
    wing.dihedral                = 8.4 * Units.degrees
    wing.origin                  = [[31,0,0.44]]
    wing.vertical                = False
    wing.symmetric               = True       
    wing.high_lift               = False  
    wing                         = wing_planform(wing)
    wing.areas.exposed           = 0.9 * wing.areas.wetted 
    wing.twists.root             = 2.0 * Units.degrees
    wing.twists.tip              = 2.0 * Units.degrees    
    wing.dynamic_pressure_ratio  = 0.90

    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Vertical_Tail()
    wing.tag = 'vertical_stabilizer'
    wing.areas.reference         = 16.0
    wing.aspect_ratio            =  1.7
    wing.sweeps.quarter_chord    = 35. * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.31
    wing.dihedral                = 0.00
    wing.origin                  = [[30.4,0,1.675]]
    wing.vertical                = True
    wing.symmetric               = False       
    wing.high_lift               = False
    wing                         = wing_planform(wing)
    wing.areas.exposed           = 0.9 * wing.areas.wetted
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees    
    wing.dynamic_pressure_ratio  = 1.00
    
    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------

    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag    = 'fuselage'
    fuselage.origin = [[0,0,0]]
    fuselage.number_coach_seats    = vehicle.passengers
    fuselage.seats_abreast         = 4
    fuselage.seat_pitch            = 30. * Units.inches

    fuselage.fineness.nose         = 1.28
    fuselage.fineness.tail         = 3.48

    fuselage.lengths.nose          = 6.0
    fuselage.lengths.tail          = 9.0
    fuselage.lengths.cabin         = 21.24
    fuselage.lengths.total         = 36.24
    fuselage.lengths.fore_space    = 0.
    fuselage.lengths.aft_space     = 0.

    fuselage.width                 = 3.01 * Units.meters

    fuselage.heights.maximum       = 3.35    
    fuselage.heights.at_quarter_length          = 3.35 
    fuselage.heights.at_three_quarters_length   = 3.35 
    fuselage.heights.at_wing_root_quarter_chord = 3.35 

    fuselage.areas.side_projected  = 239.20
    fuselage.areas.wetted          = 327.01
    fuselage.areas.front_projected = 8.0110

    fuselage.effective_diameter    = 3.18

    fuselage.differential_pressure = 10**5 * Units.pascal    # Maximum differential pressure

    # add to vehicle
    vehicle.append_component(fuselage)

    # ------------------------------------------------------------------
    #  Turbofan Network
    # ------------------------------------------------------------------    


    #initialize the gas turbine network
    gt_engine                   = SUAVE.Components.Energy.Networks.Turbofan()
    gt_engine.tag               = 'turbofan'
    gt_engine.origin            = [[12.0,4.38,-2.1],[12.0,-4.38,-2.1]]
    gt_engine.number_of_engines = 2.0
    gt_engine.bypass_ratio      = 5.4
    gt_engine.engine_length     = 2.71
    gt_engine.nacelle_diameter  = 2.05
    gt_engine.inlet_diameter    = 2.0

    #compute engine areas)
    Amax                        = (np.pi/4.)*gt_engine.nacelle_diameter**2.
    Awet                        = 1.1*np.pi*gt_engine.nacelle_diameter*gt_engine.engine_length # 1.1 is simple coefficient
    #Assign engine area
    gt_engine.areas.wetted      = Awet
    #set the working fluid for the network
    working_fluid               = SUAVE.Attributes.Gases.Air()

    #add working fluid to the network
    gt_engine.working_fluid     = working_fluid


    #Component 1 : ram,  to convert freestream static to stagnation quantities
    ram           = SUAVE.Components.Energy.Converters.Ram()
    ram.tag       = 'ram'
    #add ram to the network
    gt_engine.ram = ram


    #Component 2 : inlet nozzle
    inlet_nozzle                       = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag                   = 'inlet nozzle'
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio        = 0.98
    #add inlet nozzle to the network
    gt_engine.inlet_nozzle             = inlet_nozzle


    #Component 3 :low pressure compressor    
    low_pressure_compressor                       = SUAVE.Components.Energy.Converters.Compressor()    
    low_pressure_compressor.tag                   = 'lpc'
    low_pressure_compressor.polytropic_efficiency = 0.91
    low_pressure_compressor.pressure_ratio        = 1.9    
    #add low pressure compressor to the network    
    gt_engine.low_pressure_compressor             = low_pressure_compressor

    #Component 4 :high pressure compressor  
    high_pressure_compressor                       = SUAVE.Components.Energy.Converters.Compressor()    
    high_pressure_compressor.tag                   = 'hpc'
    high_pressure_compressor.polytropic_efficiency = 0.91
    high_pressure_compressor.pressure_ratio        = 10.0   
    #add the high pressure compressor to the network    
    gt_engine.high_pressure_compressor             = high_pressure_compressor

    #Component 5 :low pressure turbine  
    low_pressure_turbine                        = SUAVE.Components.Energy.Converters.Turbine()   
    low_pressure_turbine.tag                    ='lpt'
    low_pressure_turbine.mechanical_efficiency  = 0.99
    low_pressure_turbine.polytropic_efficiency  = 0.93
    #add low pressure turbine to the network     
    gt_engine.low_pressure_turbine              = low_pressure_turbine

    #Component 5 :high pressure turbine  
    high_pressure_turbine                       = SUAVE.Components.Energy.Converters.Turbine()   
    high_pressure_turbine.tag                   ='hpt'
    high_pressure_turbine.mechanical_efficiency = 0.99
    high_pressure_turbine.polytropic_efficiency = 0.93
    #add the high pressure turbine to the network    
    gt_engine.high_pressure_turbine             = high_pressure_turbine 

    #Component 6 :combustor  
    combustor                           = SUAVE.Components.Energy.Converters.Combustor()   
    combustor.tag                       = 'Comb'
    combustor.efficiency                = 0.99 
    combustor.alphac                    = 1.0     
    combustor.turbine_inlet_temperature = 1500
    combustor.pressure_ratio            = 0.95
    combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()    
    #add the combustor to the network    
    gt_engine.combustor                 = combustor

    #Component 7 :core nozzle
    core_nozzle                       = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    core_nozzle.tag                   = 'core nozzle'
    core_nozzle.polytropic_efficiency = 0.95
    core_nozzle.pressure_ratio        = 0.99    
    #add the core nozzle to the network    
    gt_engine.core_nozzle             = core_nozzle

    #Component 8 :fan nozzle
    fan_nozzle                       = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    fan_nozzle.tag                   = 'fan nozzle'
    fan_nozzle.polytropic_efficiency = 0.95
    fan_nozzle.pressure_ratio        = 0.99
    #add the fan nozzle to the network
    gt_engine.fan_nozzle             = fan_nozzle

    #Component 9 : fan   
    fan                       = SUAVE.Components.Energy.Converters.Fan()   
    fan.tag                   = 'fan'
    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio        = 1.7  
    #add the fan to the network
    gt_engine.fan             = fan    

    #Component 10 : thrust (to compute the thrust)
    thrust     = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='compute_thrust'
    #total design thrust (includes all the engines)
    thrust.total_design   = 37278.0* Units.N #Newtons

    #design sizing conditions
    altitude         = 35000.0*Units.ft
    mach_number      = 0.78 
    isa_deviation    = 0.
    # add thrust to the network
    gt_engine.thrust = thrust

    #size the turbofan
    turbofan_sizing(gt_engine,mach_number,altitude)   

    # add  gas turbine network gt_engine to the vehicle
    vehicle.append_component(gt_engine)      
    
    fuel                                  = SUAVE.Components.Physical_Component()
    vehicle.fuel                          = fuel
    fuel.mass_properties.mass             = vehicle.mass_properties.max_takeoff-vehicle.mass_properties.max_fuel
    fuel.origin                           = vehicle.wings.main_wing.mass_properties.center_of_gravity     
    fuel.mass_properties.center_of_gravity= vehicle.wings.main_wing.aerodynamic_center
    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    return vehicle

# ----------------------------------------------------------------------
#   Define the Configurations
# ---------------------------------------------------------------------

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
    config.wings['main_wing'].control_surfaces.flap.deflection  = 20. * Units.deg
    config.wings['main_wing'].control_surfaces.slat.deflection  = 25. * Units.deg
    config.V2_VS_ratio = 1.21
    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------

    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'landing'
    config.wings['main_wing'].control_surfaces.flap.deflection  = 30. * Units.deg
    config.wings['main_wing'].control_surfaces.slat.deflection  = 25. * Units.deg
    config.Vref_VS_ratio = 1.23
    configs.append(config)   

    # done!
    return configs

# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def mission_setup(analyses):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'embraer_e190ar test mission'

    # atmospheric model
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()

    # ------------------------------------------------------------------
    #   First Climb Segment
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"

    # connect vehicle configuration
    segment.analyses.extend( analyses.takeoff )

    # define segment attributes
    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 3.0   * Units.km
    segment.air_speed      = 125.0 * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Climb Segment
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_2"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.altitude_end   = 8.0   * Units.km
    segment.air_speed      = 190.0 * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Third Climb Segment
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_3"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.altitude_end = 10.668 * Units.km
    segment.air_speed    = 226.0  * Units['m/s']
    segment.climb_rate   = 3.0    * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Cruise Segment
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet

    segment.air_speed  = 447. * Units.knots 
    segment.distance   = 2100. * Units.nmi

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Descent Segment
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_1"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.altitude_end = 8.0   * Units.km
    segment.air_speed    = 220.0 * Units['m/s']
    segment.descent_rate = 4.5   * Units['m/s']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Descent Segment
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_2"

    # connect vehicle configuration
    segment.analyses.extend( analyses.landing )

    # segment attributes
    segment.altitude_end = 6.0   * Units.km
    segment.air_speed    = 195.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Third Descent Segment
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_3"

    # connect vehicle configuration
    segment.analyses.extend( analyses.landing )

    # segment attributes
    segment.altitude_end = 4.0   * Units.km
    segment.air_speed    = 170.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Fourth Descent Segment
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_4"

    segment.analyses.extend( analyses.landing )

    segment.altitude_end = 2.0   * Units.km
    segment.air_speed    = 150.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']


    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Fifth Descent Segment
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_5"

    segment.analyses.extend( analyses.landing )

    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 145.0 * Units['m/s']
    segment.descent_rate = 3.0   * Units['m/s']

    # append to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Mission definition complete
    # ------------------------------------------------------------------

    return mission

# ----------------------------------------------------------------------
#   Plot Mission
# ----------------------------------------------------------------------

def plot_mission(results,line_style='bo-'):

    # ------------------------------------------------------------------
    #   Throttle
    # ------------------------------------------------------------------
    plt.figure("Throttle History")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        eta  = results.segments[i].conditions.propulsion.throttle[:,0]
        axes.plot(time, eta, line_style)
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Throttle')
    axes.grid(True)

    # ------------------------------------------------------------------
    #   Angle of Attack
    # ------------------------------------------------------------------

    plt.figure("Angle of Attack History")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        aoa = results.segments[i].conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        axes.plot(time, aoa, line_style)
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Angle of Attack (deg)')
    axes.grid(True)

    # ------------------------------------------------------------------
    #   Fuel Burn Rate
    # ------------------------------------------------------------------
    plt.figure("Fuel Burn Rate")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mdot = results.segments[i].conditions.weights.vehicle_mass_rate[:,0]
        axes.plot(time, mdot, line_style)
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Fuel Burn Rate (kg/s)')
    axes.grid(True)

    # ------------------------------------------------------------------
    #   Altitude
    # ------------------------------------------------------------------
    plt.figure("Altitude")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        altitude = results.segments[i].conditions.freestream.altitude[:,0] / Units.km
        axes.plot(time, altitude, line_style)
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Altitude (km)')
    axes.grid(True)

    # ------------------------------------------------------------------
    #   Vehicle Mass
    # ------------------------------------------------------------------
    plt.figure("Vehicle Mass")
    axes = plt.gca()
    for i in range(len(results.segments)):
        time = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        mass = results.segments[i].conditions.weights.total_mass[:,0]
        axes.plot(time, mass, line_style)
    axes.set_xlabel('Time (mins)')
    axes.set_ylabel('Vehicle Mass (kg)')
    axes.grid(True)

    # ------------------------------------------------------------------
    #   Aerodynamics
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Forces")
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Lift   = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , Lift , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Lift (N)')
        axes.grid(True)

        axes = fig.add_subplot(3,1,2)
        axes.plot( time , Drag , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Drag (N)')
        axes.grid(True)

        axes = fig.add_subplot(3,1,3)
        axes.plot( time , Thrust , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Thrust (N)')
        axes.grid(True)

    # ------------------------------------------------------------------
    #   Aerodynamics 1
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Coefficients")
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        CLift  = segment.conditions.aerodynamics.lift_coefficient[:,0]
        CDrag  = segment.conditions.aerodynamics.drag_coefficient[:,0]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , CLift , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('CL')
        axes.grid(True)

        axes = fig.add_subplot(3,1,2)
        axes.plot( time , CDrag , line_style )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('CD')
        axes.grid(True)

        axes = fig.add_subplot(3,1,3)
        axes.plot( time , Drag   , line_style )
        axes.plot( time , Thrust , 'ro-' )
        axes.set_xlabel('Time (min)')
        axes.set_ylabel('Drag and Thrust (N)')
        axes.grid(True)


    # ------------------------------------------------------------------
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    fig = plt.figure("Drag Components")
    axes = plt.gca()
    for i, segment in enumerate(results.segments.values()):

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        drag_breakdown = segment.conditions.aerodynamics.drag_breakdown
        cdp = drag_breakdown.parasite.total[:,0]
        cdi = drag_breakdown.induced.total[:,0]
        cdc = drag_breakdown.compressible.total[:,0]
        cdm = drag_breakdown.miscellaneous.total[:,0]
        cd  = drag_breakdown.total[:,0]

        if line_style == 'bo-':
            axes.plot( time , cdp , 'ko-', label='CD_P' )
            axes.plot( time , cdi , 'bo-', label='CD_I' )
            axes.plot( time , cdc , 'go-', label='CD_C' )
            axes.plot( time , cdm , 'yo-', label='CD_M' )
            axes.plot( time , cd  , 'ro-', label='CD'   )
            if i == 0:
                axes.legend(loc='upper center')
        else:
            axes.plot( time , cdp , line_style )
            axes.plot( time , cdi , line_style )
            axes.plot( time , cdc , line_style )
            axes.plot( time , cdm , line_style )
            axes.plot( time , cd  , line_style )

    axes.set_xlabel('Time (min)')
    axes.set_ylabel('CD')
    axes.grid(True)

    return


if __name__ == '__main__':
    main()
    plt.show()