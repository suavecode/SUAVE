# tut_Concorde.py
# 
# Created:  Aug 2014, SUAVE Team
# Modified: Jan 2017, T. MacDonald
#           Aug 2017, E. Botero

""" setup file for a mission with a concorde
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Methods.Propulsion.turbojet_sizing import turbojet_sizing

import numpy as np
import pylab as plt

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

    plot_mission(results)
    
    plt.show()
    
    for i, segment in enumerate(results.segments.values()):
        print(segment.state.unknowns.area_initial_freestream)

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
    mission  = mission_setup(configs_analyses, vehicle)
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
    aerodynamics = SUAVE.Analyses.Aerodynamics.Supersonic_Zero()
    aerodynamics.geometry = vehicle    
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Stability Analysis
    
    # Not yet available for this configuration

    # ------------------------------------------------------------------
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.propulsors #what is called throughout the mission (at every time step))
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

    # done!
    return analyses    

def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Concorde'    
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff     = 185000. * Units.kilogram   
    vehicle.mass_properties.operating_empty = 78700.  * Units.kilogram   
    vehicle.mass_properties.takeoff         = 185000. * Units.kilogram   
    vehicle.mass_properties.cargo           = 1000.   * Units.kilogram   
        
    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 358.25 * Units['meter**2']  
    vehicle.passengers             = 100
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "long range"
    
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    
    wing.aspect_ratio            = 1.83
    wing.sweeps.quarter_chord    = 59.5 * Units.deg
    wing.thickness_to_chord      = 0.03
    wing.taper                   = 0.
    wing.span_efficiency         = .8
    wing.spans.projected         = 25.6 * Units.meter
    wing.chords.root             = 33.8 * Units.meter
    wing.total_length            = 33.8 * Units.meter
    wing.chords.tip              = 1.1  * Units.meter
    wing.chords.mean_aerodynamic = 18.4 * Units.meter
    wing.areas.reference         = 358.25 * Units['meter**2']  
    wing.areas.wetted            = 653. - 12.*2.4*2 # 2.4 is engine area on one side
    wing.areas.exposed           = 326.5  * Units['meter**2']  
    wing.areas.affected          = .6 * wing.areas.reference
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    wing.origin                  = [14,0,-.8] # meters
    wing.aerodynamic_center      = [35,0,0] # meters
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True
    wing.vortex_lift             = True
    wing.high_mach               = True
    wing.dynamic_pressure_ratio  = 1.0 
    
    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'    
    
    wing.aspect_ratio            = 0.74   
    wing.sweeps.quarter_chord    = 60 * Units.deg
    wing.thickness_to_chord      = 0.04
    wing.taper                   = 0.14
    wing.span_efficiency         = 0.9
    wing.spans.projected         = 6.0   * Units.meter   
    wing.chords.root             = 14.5  * Units.meter
    wing.total_length            = 14.5  * Units.meter
    wing.chords.tip              = 2.7   * Units.meter
    wing.chords.mean_aerodynamic = 8.66  * Units.meter
    wing.areas.reference         = 33.91 * Units['meter**2']  
    wing.areas.wetted            = 76.   * Units['meter**2']  
    wing.areas.exposed           = 38.   * Units['meter**2']  
    wing.areas.affected          = 33.91 * Units['meter**2']  
    wing.twists.root             = 0.0   * Units.degrees
    wing.twists.tip              = 0.0   * Units.degrees  
    wing.origin                  = [42.,0,1.] # meters
    wing.aerodynamic_center      = [50,0,0] # meters
    wing.vertical                = True 
    wing.symmetric               = False
    wing.t_tail                  = False
    wing.high_mach               = True     
    wing.dynamic_pressure_ratio  = 1.0
    
    # add to vehicle
    vehicle.append_component(wing)    

    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    fuselage.seats_abreast         = 4
    fuselage.seat_pitch            = 1     * Units.meter
    fuselage.fineness.nose         = 4.3   * Units.meter   
    fuselage.fineness.tail         = 6.4   * Units.meter   
    fuselage.lengths.total         = 61.66 * Units.meter    
    fuselage.width                 = 2.88  * Units.meter   
    fuselage.heights.maximum       = 3.32  * Units.meter   
    fuselage.heights.maximum       = 3.32  * Units.meter   
    fuselage.heights.at_quarter_length          = 3.32 * Units.meter   
    fuselage.heights.at_wing_root_quarter_chord = 3.32 * Units.meter   
    fuselage.heights.at_three_quarters_length   = 3.32 * Units.meter   
    fuselage.areas.wetted          = 447. * Units['meter**2'] 
    fuselage.areas.front_projected = 11.9 * Units['meter**2'] 
    fuselage.effective_diameter    = 3.1 * Units.meter    
    fuselage.differential_pressure = 7.4e4 * Units.pascal    # Maximum differential pressure
    
    # add to vehicle
    vehicle.append_component(fuselage)
         
    # ------------------------------------------------------------------
    #   Turbojet Network
    # ------------------------------------------------------------------    
    
    # instantiate the gas turbine network
    turbojet = SUAVE.Components.Energy.Networks.Turbojet_Super_Inlet()
    turbojet.tag = 'turbojet'
    
    # setup
    turbojet.number_of_engines = 4.0
    turbojet.engine_length     = 12.0
    turbojet.nacelle_diameter  = 1.3 * Units.meter
    turbojet.inlet_diameter    = 1.1 * Units.meter
    turbojet.areas             = Data()
    turbojet.areas.wetted      = 12.5*4.7*2. * Units['meter**2']  # 4.7 is outer perimeter on one side
    turbojet.origin            = [[37.,6.,-1.3],[37.,5.3,-1.3],[37.,-5.3,-1.3],[37.,-6.,-1.3]] # meters
    
    # working fluid
    turbojet.working_fluid = SUAVE.Attributes.Gases.Air()

    # ------------------------------------------------------------------
    #   Component 1 - Ram
    
    # to convert freestream static to stagnation quantities
    
    # instantiate
    ram = SUAVE.Components.Energy.Converters.Ram()
    ram.tag = 'ram'
    
    # add to the network
    turbojet.append(ram)

    # ------------------------------------------------------------------
    #  Component 2 - Inlet Nozzle
    
    inlet_nozzle = SUAVE.Components.Energy.Converters.Axisymmetric_Inlet()
    inlet_nozzle.tag = 'inlet_nozzle'
    
    # setup
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio        = 0.98
    inlet_nozzle.areas.capture         = 1.1/2.**2*np.pi* Units['meter**2'] 
    inlet_nozzle.areas.throat          = 1./2.**2*np.pi* Units['meter**2'] 
    inlet_nozzle.areas.inlet_entrance  = 1.1/2.**2*np.pi* Units['meter**2']  # 4.7 is outer perimeter on one side
    inlet_nozzle.areas.drag_direct_projection = 1/20*12.5*4.7 * Units['meter**2'] 
    inlet_nozzle.angles.cone_half_angle     = 10.0 * Units.deg
    
    # add to network
    turbojet.append(inlet_nozzle)
    
#    # instantiate
#    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
#    inlet_nozzle.tag = 'inlet_nozzle'
#    
#    # setup
#    inlet_nozzle.polytropic_efficiency = 0.98
#    inlet_nozzle.pressure_ratio        = 1.0
#    
#    # add to network
#    turbojet.append(inlet_nozzle)
    
    # ------------------------------------------------------------------
    #  Component 3 - Low Pressure Compressor
    
    # instantiate 
    compressor = SUAVE.Components.Energy.Converters.Compressor()    
    compressor.tag = 'low_pressure_compressor'

    # setup
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio        = 3.1    
    
    # add to network
    turbojet.append(compressor)

    # ------------------------------------------------------------------
    #  Component 4 - High Pressure Compressor
    
    # instantiate
    compressor = SUAVE.Components.Energy.Converters.Compressor()    
    compressor.tag = 'high_pressure_compressor'
    
    # setup
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio        = 5.0  
    
    # add to network
    turbojet.append(compressor)

    # ------------------------------------------------------------------
    #  Component 5 - Low Pressure Turbine
    
    # instantiate
    turbine = SUAVE.Components.Energy.Converters.Turbine()   
    turbine.tag='low_pressure_turbine'
    
    # setup
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.93     
    
    # add to network
    turbojet.append(turbine)
    
    # ------------------------------------------------------------------
    #  Component 6 - High Pressure Turbine
    
    # instantiate
    turbine = SUAVE.Components.Energy.Converters.Turbine()   
    turbine.tag='high_pressure_turbine'

    # setup
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.93     
    
    # add to network
    turbojet.append(turbine)
      
    # ------------------------------------------------------------------
    #  Component 7 - Combustor
    
    # instantiate    
    combustor = SUAVE.Components.Energy.Converters.Combustor()   
    combustor.tag = 'combustor'
    
    # setup
    combustor.efficiency                = 0.99   
    combustor.turbine_inlet_temperature = 1450.
    combustor.pressure_ratio            = 1.0
    combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()    
    
    # add to network
    turbojet.append(combustor)

    # ------------------------------------------------------------------
    #  Component 8 - Core Nozzle
    
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.Supersonic_Nozzle()   
    nozzle.tag = 'core_nozzle'
    
    # setup
    nozzle.polytropic_efficiency = 0.95
    nozzle.pressure_ratio        = 0.99    
    
    # add to network
    turbojet.append(nozzle)

    
    # ------------------------------------------------------------------
    #Component 9 : Thrust (to compute the thrust)
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='compute_thrust'
 
    # total design thrust (includes all the engines)
    thrust.total_design             = 4*140000. * Units.N #Newtons
 
    # Note: Sizing builds the propulsor. It does not actually set the size of the turbojet
    # design sizing conditions
    altitude      = 0.0*Units.ft
    mach_number   = 0.01
    isa_deviation = 0.
    
    altitude      = 18.288   * Units.km
    mach_number   = 2.02
    isa_deviation = 0.
    
    # add to network
    turbojet.thrust = thrust

    #size the turbojet
    turbojet_sizing(turbojet,mach_number,altitude, inlet_drag = True)   
    
    # add  gas turbine network gt_engine to the vehicle
    vehicle.append_component(turbojet)      
    
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
    
    config.wings['main_wing'].flaps.angle = 0. * Units.deg
    config.wings['main_wing'].slats.angle = 0. * Units.deg
    
    config.V2_VS_ratio = 1.21
    config.maximum_lift_coefficient = 2.
    
    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------

    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'landing'
    
    config.wings['main_wing'].flaps_angle = 0. * Units.deg
    config.wings['main_wing'].slats_angle = 0. * Units.deg

    config.Vref_VS_ratio = 1.23
    config.maximum_lift_coefficient = 2.
    
    configs.append(config)
    
    return configs

# ----------------------------------------------------------------------
#   Plot Mission
# ----------------------------------------------------------------------

def plot_mission(results,line_style='bo-'):

    axis_font = {'fontname':'Arial', 'size':'14'}    

    # ------------------------------------------------------------------
    #   Propulsion
    # ------------------------------------------------------------------


    fig = plt.figure("Propulsion",figsize=(8,6))
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0] /Units.lbf
        eta  = segment.conditions.propulsion.throttle[:,0]

        axes = fig.add_subplot(2,1,1)
        axes.plot( time , Thrust , line_style )
        axes.set_ylabel('Thrust (lbf)',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(2,1,2)
        axes.plot( time , eta , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('eta (lb/lbf-hr)',axis_font)
        axes.grid(True)	

    # ------------------------------------------------------------------
    #   Aerodynamics
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

    # ------------------------------------------------------------------
    #   Drag
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

    # ------------------------------------------------------------------
    #   Altitude, Vehicle Weight, Mach Number
    # ------------------------------------------------------------------

    fig = plt.figure("Altitude_sfc_weight",figsize=(8,10))
    for segment in results.segments.values():

        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        mass     = segment.conditions.weights.total_mass[:,0] / Units.lb
        altitude = segment.conditions.freestream.altitude[:,0] / Units.feet
        mach     = segment.conditions.freestream.mach_number[:,0]

        axes = fig.add_subplot(3,1,1)
        axes.plot( time , altitude , line_style )
        axes.set_ylabel('Altitude (ft)',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(3,1,2)
        axes.plot( time , mass , 'ro-' )
        axes.set_ylabel('Weight (lb)',axis_font)
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,3)
        axes.plot( time , mach , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Mach Number',axis_font)
        axes.grid(True)        

    return

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
        
    # fuselage seats
    base.fuselages['fuselage'].number_coach_seats = base.passengers

    # diff the new data
    base.store_diff()


    # done!
    return

# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------

def mission_setup(analyses, vehicle):
    
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
    
    ones_row     = base_segment.state.ones_row
    
    base_segment.state.unknowns.area_initial_freestream      = ones_row(1) * 1.1/2.**2*np.pi* Units['meter**2'] 
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.turbojet.unpack_unknowns
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.turbojet.residuals    
    base_segment.state.residuals.network                     = 0. * ones_row(1)  
    
    
#    # ------------------------------------------------------------------
#    #   First Climb Segment
#    # ------------------------------------------------------------------
#    
#    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
#    segment.tag = "climb_1"
#    
#    segment.analyses.extend( analyses.base )
#    
#    ones_row = segment.state.ones_row
#    segment.state.unknowns.body_angle = ones_row(1) * 7. * Units.deg   
#    
#    segment.altitude_start = 0.0   * Units.km
#    segment.altitude_end   = 3.05   * Units.km
#    segment.air_speed      = 128.6 * Units['m/s']
#    segment.climb_rate     = 20.32 * Units['m/s']
#    
#    # add to misison
#    mission.append_segment(segment)
#    
#    
#    # ------------------------------------------------------------------
#    #   Second Climb Segment
#    # ------------------------------------------------------------------    
#    
#    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
#    segment.tag = "climb_2"
#    
#    segment.analyses.extend( analyses.base )
#    
#    segment.altitude_end   = 4.57   * Units.km
#    segment.air_speed      = 205.8  * Units['m/s']
#    segment.climb_rate     = 10.16  * Units['m/s']
#    
#    # add to mission
#    mission.append_segment(segment)
#    
#    
#    # ------------------------------------------------------------------
#    #   Third Climb Segment: linear Mach
#    # ------------------------------------------------------------------    
#    
#    segment = Segments.Climb.Linear_Mach_Constant_Rate(base_segment)
#    segment.tag = "climb_3"
#    
#    segment.analyses.extend( analyses.base )
#    
#    segment.altitude_end = 7.60   * Units.km
#    segment.mach_start   = 0.64
#    segment.mach_end     = 1.0
#    segment.climb_rate   = 5.05  * Units['m/s']
#    
#    # add to mission
#    mission.append_segment(segment)
#    
#    # ------------------------------------------------------------------
#    #   Fourth Climb Segment: linear Mach
#    # ------------------------------------------------------------------    
#    
#    segment = Segments.Climb.Linear_Mach_Constant_Rate(base_segment)
#    segment.tag = "climb_4"
#    
#    segment.analyses.extend( analyses.base )
#    
#    segment.altitude_end = 15.24   * Units.km
#    segment.mach_start   = 1.0
#    segment.mach_end     = 2.02
#    segment.climb_rate   = 5.08  * Units['m/s']
#    
#    # add to mission
#    mission.append_segment(segment)
#    
#
#    # ------------------------------------------------------------------
#    #   Fourth Climb Segment
#    # ------------------------------------------------------------------    
#
#    segment = Segments.Climb.Constant_Mach_Constant_Rate(base_segment)
#    segment.tag = "climb_5"
#    
#    segment.analyses.extend( analyses.base )
#    
#    segment.altitude_end = 18.288   * Units.km
#    segment.mach_number  = 2.02
#    segment.climb_rate   = 0.65  * Units['m/s']
#    
#    # add to mission
#    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Cruise Segment
    # ------------------------------------------------------------------    
    
    segment = Segments.Cruise.Constant_Mach_Constant_Altitude(base_segment)
    segment.tag = "cruise"
    #Segments.Single_Point.Set_Speed_Set_Altitude(base_segment)
    
    segment.analyses.extend( analyses.base )
    
    segment.mach       = 2.02
    segment.distance   = 2000.0 * Units.km
    segment.altitude   = 18.288   * Units.km
        
    mission.append_segment(segment)
#    
#    # ------------------------------------------------------------------    
#    #   First Descent Segment
#    # ------------------------------------------------------------------    
#    
#    segment = Segments.Descent.Linear_Mach_Constant_Rate(base_segment)
#    segment.tag = "descent_1"
#    
#    segment.analyses.extend( analyses.base )
#    
#    segment.altitude_end = 6.8   * Units.km
#    segment.mach_start   = 2.02
#    segment.mach_end     = 1.0
#    segment.descent_rate = 5.0   * Units['m/s']
#    
#    # add to mission
#    mission.append_segment(segment)
#    
#    # ------------------------------------------------------------------    
#    #   Second Descent Segment
#    # ------------------------------------------------------------------    
#    
#    segment = Segments.Descent.Linear_Mach_Constant_Rate(base_segment)
#    segment.tag = "descent_2"
#    
#    segment.analyses.extend( analyses.base )
#    
#    segment.altitude_end = 3.0   * Units.km
#    segment.mach_start   = 1.0
#    segment.mach_end     = 0.65
#    segment.descent_rate = 5.0   * Units['m/s']
#    
#    # add to mission
#    mission.append_segment(segment)    
#    
#    # ------------------------------------------------------------------    
#    #   Third Descent Segment
#    # ------------------------------------------------------------------    
#
#    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
#    segment.tag = "descent_3"
#
#    segment.analyses.extend( analyses.base )
#    
#    segment.altitude_end = 0.0   * Units.km
#    segment.air_speed    = 130.0 * Units['m/s']
#    segment.descent_rate = 5.0   * Units['m/s']
#
#    # append to mission
#    mission.append_segment(segment)
#    
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

    # done!
    return missions  

if __name__ == '__main__': 
    
    main()