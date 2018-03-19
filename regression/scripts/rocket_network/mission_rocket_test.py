# mission_rocket_test.py
# 
# Created:  March 2018, W. Maier
# Modified: 

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

# Python Imports
import numpy as np
import pylab as plt

# SUAVE Imports
import SUAVE
from SUAVE.Core import Data, Units
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Propulsion import compute_turbofan_geometry
from SUAVE.Input_Output.Results import  print_parasite_drag,  \
    print_compress_drag, \
    print_engine_data,   \
    print_mission_breakdown, \
    print_weight_breakdown

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    configs, analyses = full_setup()
    simple_sizing(configs)
    configs.finalize()
    analyses.finalize()

    # weight analysis
    weights = analyses.configs.base.weights
    breakdown = weights.evaluate()      

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()

    # print weight breakdown
    print_weight_breakdown(configs.base,filename = 'B737_weight_breakdown.dat')

    # print engine data into file
    print_engine_data(configs.base,filename = 'B737_engine_data.dat')

    # print parasite drag data into file
    # define reference condition for parasite drag
    ref_condition = Data()
    ref_condition.mach_number = 0.3
    ref_condition.reynolds_number = 12e6     
    print_parasite_drag(ref_condition,configs.cruise,analyses,'B737_parasite_drag.dat')

    # print compressibility drag data into file
    print_compress_drag(configs.cruise,analyses,filename = 'B737_compress_drag.dat')

    # print mission breakdown
    print_mission_breakdown(results,filename='B737_mission_breakdown.dat')

    # plt the old results
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
    missions_analyses = missions_setup(mission)
    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

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
    weights = SUAVE.Analyses.Weights.Weights_Tube_Wing()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Stability Analysis
    stability = SUAVE.Analyses.Stability.Fidelity_Zero()
    stability.geometry = vehicle
    analyses.append(stability)

    # ------------------------------------------------------------------
    #  Energy
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
    vehicle.tag = 'Boeing_737-800'    

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff               = 79015.8 * Units.kilogram 
    vehicle.mass_properties.takeoff                   = 79015.8 * Units.kilogram   
    vehicle.mass_properties.operating_empty           = 62746.4 * Units.kilogram 
    vehicle.mass_properties.takeoff                   = 79015.8 * Units.kilogram 
    vehicle.mass_properties.max_zero_fuel             = 62732.0 * Units.kilogram 
    vehicle.mass_properties.cargo                     = 10000.  * Units.kilogram   

    # envelope properties
    vehicle.envelope.ultimate_load = 2.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 124.862 * Units['meters**2']  
    vehicle.passengers             = 170
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "medium range"

    # ------------------------------------------------------------------        
    #  Landing Gear
    # ------------------------------------------------------------------        
    # used for noise calculations
    landing_gear = SUAVE.Components.Landing_Gear.Landing_Gear()
    landing_gear.tag = "main_landing_gear"
    landing_gear.main_tire_diameter = 1.12000 * Units.m
    landing_gear.nose_tire_diameter = 0.6858 * Units.m
    landing_gear.main_strut_length  = 1.8 * Units.m
    landing_gear.nose_strut_length  = 1.3 * Units.m
    landing_gear.main_units  = 2    #number of main landing gear units
    landing_gear.nose_units  = 1    #number of nose landing gear
    landing_gear.main_wheels = 2    #number of wheels on the main landing gear
    landing_gear.nose_wheels = 2    #number of wheels on the nose landing gear      
    vehicle.landing_gear = landing_gear

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    wing.aspect_ratio            = 10.18
    wing.sweeps.quarter_chord    = 25 * Units.deg
    wing.thickness_to_chord      = 0.1
    wing.taper                   = 0.1
    wing.span_efficiency         = 0.9
    wing.spans.projected         = 34.32 * Units.meter
    wing.chords.root             = 7.760 * Units.meter
    wing.chords.tip              = 0.782 * Units.meter
    wing.chords.mean_aerodynamic = 4.235 * Units.meter
    wing.areas.reference         = 124.862 * Units['meters**2']  
    wing.twists.root             = 4.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    wing.origin                  = [13.61,0,-1.27] # meters
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True
    wing.dynamic_pressure_ratio  = 1.0

    # ------------------------------------------------------------------
    #   Flaps
    # ------------------------------------------------------------------
    wing.flaps.chord      =  0.30   # 30% of the chord
    wing.flaps.span_start =  0.10   # 10% of the span
    wing.flaps.span_end   =  0.75
    wing.flaps.type       = 'double_slotted'

    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'
    wing.aspect_ratio            = 6.16     
    wing.sweeps.quarter_chord    = 40 * Units.deg
    wing.thickness_to_chord      = 0.08
    wing.taper                   = 0.2
    wing.span_efficiency         = 0.9
    wing.spans.projected         = 14.2 * Units.meter
    wing.chords.root             = 4.7  * Units.meter
    wing.chords.tip              = .955 * Units.meter
    wing.chords.mean_aerodynamic = 8.0  * Units.meter
    wing.areas.reference         = 32.488   * Units['meters**2']  
    wing.areas.exposed           = 199.7792 * Units['meters**2']  
    wing.areas.wetted            = 249.724  * Units['meters**2']  
    wing.twists.root             = 3.0 * Units.degrees
    wing.twists.tip              = 3.0 * Units.degrees  
    wing.origin                  = [32.83,0,1.14] # meters
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
    wing.sweeps.quarter_chord    = 25. * Units.deg
    wing.thickness_to_chord      = 0.08
    wing.taper                   = 0.25
    wing.span_efficiency         = 0.9
    wing.spans.projected         = 7.777 * Units.meter
    wing.chords.root             = 8.19  * Units.meter
    wing.chords.tip              = 0.95  * Units.meter
    wing.chords.mean_aerodynamic = 4.0   * Units.meter
    wing.areas.reference         = 27.316 * Units['meters**2']  
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  
    wing.origin                  = [28.79,0,1.54] # meters
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
    fuselage.seat_pitch            = 1     * Units.meter
    fuselage.fineness.nose         = 1.6
    fuselage.fineness.tail         = 2.
    fuselage.lengths.nose          = 6.4   * Units.meter
    fuselage.lengths.tail          = 8.0   * Units.meter
    fuselage.lengths.cabin         = 28.85 * Units.meter
    fuselage.lengths.total         = 38.02 * Units.meter
    fuselage.lengths.fore_space    = 6.    * Units.meter
    fuselage.lengths.aft_space     = 5.    * Units.meter
    fuselage.width                 = 3.74  * Units.meter
    fuselage.heights.maximum       = 3.74  * Units.meter
    fuselage.effective_diameter    = 3.74     * Units.meter
    fuselage.areas.side_projected  = 142.1948 * Units['meters**2'] 
    fuselage.areas.wetted          = 446.718  * Units['meters**2'] 
    fuselage.areas.front_projected = 12.57    * Units['meters**2'] 
    fuselage.differential_pressure = 5.0e4 * Units.pascal # Maximum differential pressure
    fuselage.heights.at_quarter_length          = 3.74 * Units.meter
    fuselage.heights.at_three_quarters_length   = 3.65 * Units.meter
    fuselage.heights.at_wing_root_quarter_chord = 3.74 * Units.meter

    # add to vehicle
    vehicle.append_component(fuselage)

    # ------------------------------------------------------------------
    #   Turbofan Network
    # ------------------------------------------------------------------    
    #instantiate the gas turbine network
    turbofan = SUAVE.Components.Energy.Networks.Turbofan()
    turbofan.tag = 'turbofan'

    # setup
    turbofan.number_of_engines = 2
    turbofan.bypass_ratio      = 5.4
    turbofan.engine_length     = 2.71 * Units.meter
    turbofan.nacelle_diameter  = 2.05 * Units.meter
    turbofan.origin            = [[13.72, 4.86,-1.9],[13.72, -4.86,-1.9]] # meters

    #compute engine areas
    turbofan.areas.wetted      = 1.1*np.pi*turbofan.nacelle_diameter*turbofan.engine_length

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
    combustor.turbine_inlet_temperature = 1450 # K
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

    #Engine setup for noise module    
    # add to network
    turbofan.thrust = thrust

    #size the turbofan
    turbofan_sizing(turbofan,mach_number,altitude)   

    # add  gas turbine network turbofan to the vehicle 
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
    config.wings['main_wing'].slats.angle = 25. * Units.deg
    config.max_lift_coefficient_factor    = 1.
    configs.append(config)

    # ------------------------------------------------------------------
    #   Cutback Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cutback'
    config.wings['main_wing'].flaps.angle = 20. * Units.deg
    config.wings['main_wing'].slats.angle = 20. * Units.deg
    config.max_lift_coefficient_factor    = 1. #0.95
    configs.append(config)    

    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'landing'
    config.wings['main_wing'].flaps.angle = 30. * Units.deg
    config.wings['main_wing'].slats.angle = 25. * Units.deg  
    config.max_lift_coefficient_factor    = 1. #0.95
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

    return configs

def simple_sizing(configs):
    base = configs.base
    base.pull_base()

    # zero fuel weight
    base.mass_properties.max_zero_fuel = 0.9 * base.mass_properties.max_takeoff 

    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 2.0 * wing.areas.reference
        wing.areas.exposed  = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted

    # diff the new data
    base.store_diff()

    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------
    landing = configs.landing

    # make sure base data is current
    landing.pull_base()

    # landing weight
    landing.mass_properties.landing = 0.85 * base.mass_properties.takeoff

    # diff the new data
    landing.store_diff()

    return

# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def mission_setup(analyses):
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'the_mission'

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
    #   First Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment = Segments.Climb.Linear_Throttle_Fixed_Angle_Time(base_segment)
    segment.tag = "climb_1"
    segment.altitude_start  = 0.0   * Units.km
    segment.flight_time     = 20.0  * Units.seconds
    segment.air_speed_start = 0.0   * Units['m/s']
    segment.throttle_start  = 36.0  
    segment.throttle_end    = 40.0  

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Mission definition complete    
    # ------------------------------------------------------------------

    return mission

def missions_setup(base_mission):
    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()

    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------
    missions.base = base_mission

    return missions  

# ----------------------------------------------------------------------
#   Plot Mission
# ----------------------------------------------------------------------
def plot_mission(results,line_style='bo-'):

    axis_font = {'fontname':'Arial', 'size':'14'}    

    # ------------------------------------------------------------------
    #   Aerodynamics
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Forces",figsize=(8,6))
    for segment in results.segments.values():
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0] / Units.lbf
        eta    = segment.conditions.propulsion.throttle[:,0]
        axes = fig.add_subplot(2,1,1)
        axes.plot( time , Thrust , line_style )
        axes.set_ylabel('Thrust (lbf)',axis_font)
        axes.grid(True)
        axes = fig.add_subplot(2,1,2)
        axes.plot( time , eta , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Throttle',axis_font)
        axes.grid(True)	
        plt.savefig("B737_engine.pdf")
        plt.savefig("B737_engine.png")

    # ------------------------------------------------------------------
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    fig = plt.figure("Aerodynamic Coefficients",figsize=(8,10))
    for segment in results.segments.values():
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        CLift  = segment.conditions.aerodynamics.lift_coefficient[:,0]
        CDrag  = segment.conditions.aerodynamics.drag_coefficient[:,0]
        aoa = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        l_d = CLift/CDrag
        axes = fig.add_subplot(3,1,1)
        axes.plot( time , CLift , line_style )
        axes.set_ylabel('Lift Coefficient',axis_font)
        axes.grid(True)
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , l_d , line_style )
        axes.set_ylabel('L/D',axis_font)
        axes.grid(True)
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , aoa , 'ro-' )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('AOA (deg)',axis_font)
        axes.grid(True)
        plt.savefig("B737_aero.pdf")
        plt.savefig("B737_aero.png")

    # ------------------------------------------------------------------
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    fig = plt.figure("Drag Components",figsize=(8,10))
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
            axes.plot( time , cdp , 'ko-', label='CD parasite' )
            axes.plot( time , cdi , 'bo-', label='CD induced' )
            axes.plot( time , cdc , 'go-', label='CD compressibility' )
            axes.plot( time , cdm , 'yo-', label='CD miscellaneous' )
            axes.plot( time , cd  , 'ro-', label='CD total'   )

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
    plt.savefig("B737_drag.pdf")
    plt.savefig("B737_drag.png")

    # ------------------------------------------------------------------
    #   Altitude, sfc, vehicle weight
    # ------------------------------------------------------------------
    fig = plt.figure("Altitude_sfc_weight",figsize=(8,10))
    for segment in results.segments.values():
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        aoa      = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        mass     = segment.conditions.weights.total_mass[:,0] / Units.lb
        altitude = segment.conditions.freestream.altitude[:,0] / Units.ft
        mdot     = segment.conditions.weights.vehicle_mass_rate[:,0]
        thrust   =  segment.conditions.frames.body.thrust_force_vector[:,0]
        sfc      = (mdot / Units.lb) / (thrust /Units.lbf) * Units.hr
        axes = fig.add_subplot(3,1,1)
        axes.plot( time , altitude , line_style )
        axes.set_ylabel('Altitude (ft)',axis_font)
        axes.grid(True)
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , sfc , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('sfc (lb/lbf-hr)',axis_font)
        axes.grid(True)
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , mass , 'ro-' )
        axes.set_ylabel('Weight (lb)',axis_font)
        axes.grid(True)

        plt.savefig("B737_mission.pdf")
        plt.savefig("B737_mission.png")

    # ------------------------------------------------------------------
    #   Velocities
    # ------------------------------------------------------------------
    fig = plt.figure("Velocities",figsize=(8,10))
    for segment in results.segments.values():
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        Lift     = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag     = -segment.conditions.frames.wind.drag_force_vector[:,0] / Units.lbf
        Thrust   = segment.conditions.frames.body.thrust_force_vector[:,0] / Units.lb
        velocity = segment.conditions.freestream.velocity[:,0]
        pressure = segment.conditions.freestream.pressure[:,0]
        density  = segment.conditions.freestream.density[:,0]
        EAS      = velocity * np.sqrt(density/1.225)
        mach     = segment.conditions.freestream.mach_number[:,0]

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , velocity / Units.kts, line_style )
        axes.set_ylabel('velocity (kts)',axis_font)
        axes.grid(True)
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , EAS / Units.kts, line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Equivalent Airspeed',axis_font)
        axes.grid(True)    

        axes = fig.add_subplot(3,1,3)
        axes.plot( time , mach , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Mach',axis_font)
        axes.grid(True)           

    return

if __name__ == '__main__': 

    main()    
    plt.show()