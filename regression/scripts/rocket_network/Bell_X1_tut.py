# Bell_X1_tut.py
# 
# Created:  April 2018, W. Maier
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
from SUAVE.Methods.Propulsion.liquid_rocket_sizing import liquid_rocket_sizing
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

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()

    # plt the results
    plot_mission(results)
    plt.show()

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
    aerodynamics = SUAVE.Analyses.Aerodynamics.Supersonic_Zero()
    aerodynamics.geometry = vehicle
    analyses.append(aerodynamics)

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
    vehicle.tag = 'Bell_X-1'    

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff               = 12250.0 * Units.pounds 
    vehicle.mass_properties.takeoff                   = 12250.0 * Units.pounds   
    vehicle.mass_properties.operating_empty           = 7150.0  * Units.pounds 
    vehicle.mass_properties.max_zero_fuel             = 7150.0  * Units.pounds 
    vehicle.mass_properties.cargo                     = 150.0   * Units.pounds  

    # envelope properties
    vehicle.envelope.ultimate_load = 2.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 130.0 * Units['feet**2']  
    vehicle.passengers             = 0
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "medium range"

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    wing.aspect_ratio            = 6.0308
    wing.sweeps.quarter_chord    = 0 * Units.deg
    wing.thickness_to_chord      = 0.08
    wing.taper                   = 0.4979  
    wing.span_efficiency         = 0.9  
    wing.spans.projected         = 28.0   * Units.feet
    wing.chords.root             = 6.183  * Units.feet
    wing.chords.tip              = 3.789  * Units.feet
    wing.chords.mean_aerodynamic = 4.8045 * Units.feet
    wing.total_length            = 6.183  * Units.feet
    wing.areas.reference         = 130.   * Units['feet**2'] 
    wing.areas.exposed           = 221.82 * Units['feet**2']  
    wing.areas.wetted            = 260.0  * Units['feet**2']      
    wing.twists.root             = 0.0    * Units.degrees
    wing.twists.tip              = 0.0    * Units.degrees
    wing.origin                  = [12.67*Units.ft,0*Units.ft,0*Units.feet]
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False
    wing.dynamic_pressure_ratio  = 1.0

    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'
    wing.aspect_ratio            = 5.1147   
    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.06
    wing.taper                   = 0.4884
    wing.span_efficiency         = 0.9
    wing.spans.projected         = 14.2 * Units.feet
    wing.chords.root             = 2.99 * Units.feet
    wing.chords.tip              = 1.46 * Units.feet
    wing.chords.mean_aerodynamic = 2.31 * Units.feet
    wing.total_length            = 2.99 * Units.feet
    wing.areas.reference         = 25.3279  * Units['feet**2'] 
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  
    wing.origin                  = [27.07*Units.ft,0*Units.ft,3.52*Units.feet] 
    wing.vertical                = False 
    wing.symmetric               = True
    wing.dynamic_pressure_ratio  = 1.0  

    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'    
    wing.aspect_ratio            = 1.9528
    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.08
    wing.taper                   = 0.5410
    wing.span_efficiency         = 0.9
    wing.spans.projected         = 8.027  * Units.feet
    wing.chords.root             = 5.335  * Units.feet
    wing.chords.tip              = 2.885  * Units.feet
    wing.chords.mean_aerodynamic = 4.232  * Units.feet
    wing.total_length            = 5.335  * Units.feet
    wing.areas.reference         = 32.995 * Units['feet**2']  
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  
    wing.origin                  = [30.04*Units.ft,0*Units.ft,1.0*Units.feet] # feet??
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
    fuselage.fineness.nose         = 2.7785
    fuselage.fineness.tail         = 4.0011
    fuselage.lengths.nose          = 12.67 * Units.feet
    fuselage.lengths.tail          = 18.245* Units.feet
    fuselage.lengths.cabin         = 0.0   * Units.feet
    fuselage.lengths.total         = 30.915* Units.feet
    
    fuselage.width                 = 4.56  * Units.feet
    fuselage.heights.maximum       = 4.56  * Units.feet
    fuselage.effective_diameter    = 4.56  * Units.feet
    
    #-----
    fuselage.areas.side_projected               = 131.3887 * Units['feet**2'] 
    fuselage.areas.wetted                       = 411.55   * Units['feet**2']   
    fuselage.areas.front_projected              = 16.33    * Units['feet**2'] 
    fuselage.differential_pressure              = 10.0e4   * Units.pascal # Maximum differential pressure
    fuselage.heights.at_quarter_length          = 6.055    * Units.feet
    fuselage.heights.at_three_quarters_length   = 5.167    * Units.feet
    fuselage.heights.at_wing_root_quarter_chord = 5.74     * Units.feet
    # ------
    
    # add to vehicle
    vehicle.append_component(fuselage)

    # ------------------------------------------------------------------
    #   Rocket Network
    # ------------------------------------------------------------------    
    #instantiate the gas turbine network
    liquid_rocket = SUAVE.Components.Energy.Networks.Liquid_Rocket()
    liquid_rocket.tag = 'liquid_rocket'
    
    # Areas are zero, rocket is internal
    liquid_rocket.engine_length     = 0.0 * Units.meter
    liquid_rocket.nacelle_diameter  = 0.0 * Units.meter
    liquid_rocket.areas.wetted      = 1.*np.pi*liquid_rocket.nacelle_diameter*liquid_rocket.engine_length
    
    # setup
    liquid_rocket.number_of_engines = 4  #In reality it is one rocket, with four chambers
    liquid_rocket.origin            = [[30.915*Units.ft,.5*Units.ft,-.5*Units.ft],[30.915*Units.ft,.5*Units.ft,.5*Units.ft],[330.915*Units.ft,-.5*Units.ft,-.5*Units.ft],[30.915*Units.ft,-.5*Units.ft,.5*Units.ft]]
    # working fluid
    liquid_rocket.working_fluid = SUAVE.Attributes.Gases.Air()

    # ------------------------------------------------------------------
    #   Component 1 - Combustor
    # instantiate
    combustor = SUAVE.Components.Energy.Converters.Rocket_Combustor()
    combustor.tag = 'combustor'

    # setup  
    combustor.propellant_data                = SUAVE.Attributes.Propellants.LOX_Ethyl()
    combustor.inputs.combustion_pressure     = 1823850.0     
    
    # add to the network
    liquid_rocket.append(combustor)

    # ------------------------------------------------------------------
    #  Component 2 - Nozzle
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.de_Laval_Nozzle()
    nozzle.tag = 'core_nozzle'

    # setup
    nozzle.polytropic_efficiency = 0.98
    nozzle.pressure_ratio        = 0.98
    nozzle.expansion_ratio       = 6.3434
    nozzle.area_throat           = 0.0029 *Units.meter
    
    # add to network
    liquid_rocket.append(nozzle)

    # ------------------------------------------------------------------
    #Component 3 : thrust (to compute the thrust)
    thrust = SUAVE.Components.Energy.Processes.Rocket_Thrust()       
    thrust.tag ='compute_thrust'

    #total design thrust (includes all the engines)
    thrust.total_design   = 4*7043.8 * Units.N #Newtons
    thrust.ISP_design     = 263.4193
    
    #design sizing conditions
    altitude      = 0.0 *Units.feet
  
    # add to network
    liquid_rocket.thrust = thrust

    #size the liquid_rocket
    liquid_rocket_sizing(liquid_rocket,altitude)   

    # add rocket to the vehicle 
    vehicle.append_component(liquid_rocket)      

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

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()

    # ------------------------------------------------------------------
    # Climb Segment: Constant Throttle, Constant Speed
    # ------------------------------------------------------------------ 
    #segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment = Segments.Climb.Constant_Throttle_Constant_Speed(base_segment)
    
    segment.tag = "climb_self"
    segment.analyses.extend(analyses.cruise)
    segment.altitude_start  = 23000.0 * Units.feet
    segment.altitude_end    = 43000.0 * Units.feet
    segment.throttle        = 1.00
   # segment.climb_rate      = 14000   * Units.ft/Units.minute
    segment.air_speed       = 150.0   * Units.m/Units.s
    segment.state.numerics.number_control_points = 32

    # add to misison
    mission.append_segment(segment)    

    # ------------------------------------------------------------------
    # Cruise Segment: Constant Throttle, Constant Altitude
    # ------------------------------------------------------------------
    #segment = Segments.Cruise.Constant_Throttle_Constant_Altitude(base_segment)
    #segment.tag = "cruise"
    #segment.analyses.extend(analyses.cruise)    
    #segment.altitude        = 43000.0 * Units.feet
    #segment.throttle        = 1.0
    #segment.air_speed_start = 150.0   * Units.m/Units.s
    #segment.air_speed_end   = 316.0   * Units['m/s']
    #segment.state.numerics.number_control_points = 200   

    # add to misison
    #mission.append_segment(segment)
  
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
        #plt.savefig("B737_engine.pdf")
        #plt.savefig("B737_engine.png")

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
        axes.get_yaxis().get_major_formatter().set_scientific(False)        
        axes.grid(True)
        
        axes = fig.add_subplot(3,1,2)
        axes.plot( time , l_d , line_style )
        axes.set_ylabel('L/D',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.grid(True)

        axes = fig.add_subplot(3,1,3)
        axes.plot( time , aoa , 'ro-' )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('AOA (deg)',axis_font)
        axes.get_yaxis().get_major_formatter().set_scientific(False)
        axes.grid(True)
        #plt.savefig("B737_aero.pdf")
        #plt.savefig("B737_aero.png")

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
    #plt.savefig("B737_drag.pdf")
    #plt.savefig("B737_drag.png")

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

        #plt.savefig("B737_mission.pdf")
        #plt.savefig("B737_mission.png")

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