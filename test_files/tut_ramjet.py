# tut_Concorde.py
# 
# Created:  Aug 2014, SUAVE Team
# Modified: Jan 2017, T. MacDonald

""" setup file for a mission with a concorde
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Core import (
Data, Container,
)

from SUAVE.Methods.Propulsion.ramjet_sizing import ramjet_sizing

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main(source_ratio=1.):

    configs, analyses = full_setup(source_ratio)

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

    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup(source_ratio=1.):

    # vehicle data
    vehicle  = vehicle_setup(source_ratio)
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
    weights = SUAVE.Analyses.Weights.Weights()
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

def vehicle_setup(source_ratio=1.):

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Concorde'    
    
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff               = 185000*0.4   # kg
    vehicle.mass_properties.operating_empty           = 78700*0.4  # kg
    vehicle.mass_properties.takeoff                   = 185000*0.4   # kg
    vehicle.mass_properties.cargo                     = 0.  * Units.kilogram   
        
    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 340.25      
    vehicle.passengers             = 40
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "long range"
    
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    
    wing.aspect_ratio            = 1.83
    wing.sweeps.quarter_chord     = 59.5 * Units.deg
    wing.thickness_to_chord      = 0.03
    wing.taper                   = 0.
    wing.span_efficiency         = 0.74
    
    wing.spans.projected         = 25.6    
    
    wing.chords.root             = 33.8
    wing.total_length            = 33.8
    wing.chords.tip              = 1.1
    wing.chords.mean_aerodynamic = 18.4
    
    wing.areas.reference         = 358.25 
    
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    
    wing.origin                  = [14,0,-.8]
    wing.aerodynamic_center      = [35,0,0] 
    
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
    
    wing.aspect_ratio            = 0.74      #
    wing.sweeps.quarter_chord    = 60 * Units.deg
    wing.thickness_to_chord      = 0.04
    wing.taper                   = 0.14
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 6.0      #    

    wing.chords.root             = 14.5
    wing.total_length            = 14.5
    wing.chords.tip              = 2.7
    wing.chords.mean_aerodynamic = 8.66
    
    wing.areas.reference         = 33.91    #
    
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  
    
    wing.origin                  = [42.,0,1.]
    wing.aerodynamic_center      = [50,0,0]    
    
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
    fuselage.seat_pitch            = 1
    
    fuselage.fineness.nose         = 4.3
    fuselage.fineness.tail         = 6.4
    
    fuselage.lengths.total         = 61.66  
    
    fuselage.width                 = 2.88
    
    fuselage.heights.maximum       = 3.32    #
    
    fuselage.heights.maximum       = 3.32    #
    fuselage.heights.at_quarter_length              = 3.32    #
    fuselage.heights.at_wing_root_quarter_chord     = 3.32    #
    fuselage.heights.at_three_quarters_length       = 3.32    #

    fuselage.areas.wetted          = 523.
    fuselage.areas.front_projected = 7.55
    
    
    fuselage.effective_diameter    = 3.1
    
    fuselage.differential_pressure = 7.4e4 * Units.pascal    # Maximum differential pressure   
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
        
    # ------------------------------------------------------------------
    #   ramjet Network
    # ------------------------------------------------------------------    
    
    # instantiate the gas turbine network
    ramjet = SUAVE.Components.Energy.Networks.Ramjet()
    ramjet.tag = 'turbojet' #mudar em Methods/Aerodynamics/Drag/Compressibility_drag_total
    
    # setup
    ramjet.number_of_engines = 4.0
    ramjet.engine_length     = 12.5
    ramjet.nacelle_diameter  = 1.40
    ramjet.areas             = Data()
    ramjet.areas.wetted      = 12.5*1.4*8. # essentially rectangles attached to the wings
    
    # working fluid
    ramjet.working_fluid = SUAVE.Attributes.Gases.Air()
    
    
    # ------------------------------------------------------------------
    #   Component 1 - Ram
    
    # to convert freestream static to stagnation quantities
    
    # instantiate
    ram = SUAVE.Components.Energy.Converters.Ram()
    ram.tag = 'ram'
    
    # add to the network
    ramjet.append(ram)


    # ------------------------------------------------------------------
    #  Component 2 - Inlet Nozzle
    
    
    #MIL-E-500B
    # pressure_ratio = 1.0 -0.075(M-1)**1.35
    # instantiate
    inlet_nozzle = SUAVE.Components.Energy.Converters.Supersonic_Intake()
    inlet_nozzle.tag = 'inlet_nozzle'
    
    # setup
    inlet_nozzle.polytropic_efficiency = 1.0
    inlet_nozzle.efficiency            = 0.95
 
    
    # add to network
    ramjet.append(inlet_nozzle)
    
      
    
    # ------------------------------------------------------------------
    #  Component 7 - Combustor
    
    # instantiate    
    combustor = SUAVE.Components.Energy.Converters.Combustor()   
    combustor.tag = 'combustor'
    
    # setup
    combustor.efficiency                = 0.96
    combustor.alphac                    = 1.0     
    combustor.turbine_inlet_temperature = 2400.
    combustor.pressure_ratio            = 0.97
    combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()    
    
    # add to network
    ramjet.append(combustor)

    
    # ------------------------------------------------------------------
    #  Component 8 - Core Nozzle
    
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.Supersonic_Nozzle_V2()   
    nozzle.tag = 'core_nozzle'
    
    # setup
    nozzle.polytropic_efficiency = 0.99
    nozzle.pressure_ratio        = 0.98
    nozzle.area_ratio            = 1.5 #insert a value higher than 1.12
    
    # add to network
    ramjet.append(nozzle)
    
    
    # ------------------------------------------------------------------
    #Component 9 : thrust (to compute the thrust)
    thrust = SUAVE.Components.Energy.Processes.HyperThrust()       
    thrust.tag ='compute_thrust'
 
    # total design thrust (includes all the engines)
    thrust.total_design             =  ramjet.number_of_engines*90000. * Units.N #Newtons
 
    # Note: Sizing builds the propulsor. It does not actually set the size of the ramjet
    # Design sizing conditions
    altitude      = 12.0*Units.km
    mach_number   = 2.3
    isa_deviation = 0.
    
    # add to network
    ramjet.thrust = thrust

    #size the ramjet
    ramjet_sizing(ramjet,mach_number,altitude)   
    
    # add  gas turbine network gt_engine to the vehicle
    vehicle.append_component(ramjet)      
    
    
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


    # done!
    return configs

# ----------------------------------------------------------------------
#   Sizing for the Vehicle Configs
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
#   Plot Mission
# ----------------------------------------------------------------------

def plot_mission(results,line_style='ro-'):

    axis_font = {'fontname':'Arial', 'size':'14'}    

   # ------------------------------------------------------------------
    #   Propulsion
    # ------------------------------------------------------------------


    fig = plt.figure("Propulsion",figsize=(8,12))
    i=0
    for segment in results.segments.values():
        
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]*0.224808943
        eta  = segment.conditions.propulsion.throttle[:,0]
        mdot = segment.conditions.weights.vehicle_mass_rate[:,0]
        sfc = mdot * 7936.641 /Thrust;


        #sfc = segment.conditions.propulsion.thrust_specific_fuel_consumption


        axes = fig.add_subplot(4,1,1)
        axes.plot( time , Thrust , line_style )
        #axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Thrust (lbf)',axis_font)
        axes.grid(True)
        
        axes = fig.add_subplot(4,1,2)
        axes.plot( time , sfc , line_style)
        axes.set_ylabel('sfc (lb/lbf-hr)',axis_font)
        axes.grid(True)	

        
        axes = fig.add_subplot(4,1,3)
        axes.plot( time , eta*100 , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Throttle (%)',axis_font)
        axes.grid(True)
        
        axes = fig.add_subplot(4,1,4)
        axes.plot( time , mdot , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Fuel Burn rate (kg/s)',axis_font)
        axes.grid(True)
        
        plt.savefig("engine_1.png")
        
        #=================================================
#    fig = plt.figure("Propulsion_2",figsize=(8,9))
#    i=0
#    for segment in results.segments.values():
#        
#        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
#        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]*0.224808943
#        eta  = segment.conditions.propulsion.throttle[:,0]
#        mdot = segment.conditions.weights.vehicle_mass_rate[:,0]
#        sfc = mdot * 7936.641 /Thrust;
#        exit_velocity = segment.conditions.propulsion.exit_velocity[:,0]
#        flow_velocity = segment.conditions.freestream.velocity[:,0]
#        exit_pressure = segment.conditions.propulsion.exit_pressure[:,0]
#        freestream_pressure = segment.conditions.freestream.pressure[:,0]
#
#        axes = fig.add_subplot(3,1,1)
#        axes.plot( time , flow_velocity , 'ko-', label = 'Freestream' )
#        axes.plot( time,  exit_velocity, line_style, label = 'Engine exit')
#        if (i==0):
#            axes.legend(loc='center right')  
#            
#        axes.set_ylabel('Velocity (m/s)',axis_font)
#        axes.grid(True)	
#        
#        axes = fig.add_subplot(3,1,2)
#        axes.plot( time , freestream_pressure , 'ko-', label = 'Freestream' )
#        axes.plot( time,  exit_pressure, line_style, label = 'Engine exit')
#        axes.set_xlabel('Time (min)',axis_font)
#        axes.set_ylabel('Pressure (Pa)',axis_font)
#        axes.grid(True)
#        if (i==0):
#            axes.legend(loc='center right')  
#        
#         
#        axes = fig.add_subplot(3,1,3)
#        axes.plot( time , mdot , line_style )
#        #axes.set_xlabel('Time (min)',axis_font)
#        axes.set_ylabel('Fuel Burn rate (kg/s)',axis_font)
#        axes.grid(True)
#  
#    
#        i= i +1
#        
#        plt.savefig("Engine_2.png")



    # ------------------------------------------------------------------
    #   Altitude, Vehicle Weight, Mach Number
    # ------------------------------------------------------------------

    fig = plt.figure("Altitude_sfc_weight",figsize=(8,6))
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        mass   = segment.conditions.weights.total_mass[:,0]*2.20462
        altitude = segment.conditions.freestream.altitude[:,0] / Units.km *3.28084 *1000
        mach   = segment.conditions.freestream.mach_number[:,0]
      
    

        axes = fig.add_subplot(2,1,1)
        axes.plot( time , altitude , line_style )
        #axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Altitude (ft)',axis_font)
        axes.grid(True)
        
        axes = fig.add_subplot(2,1,2)
        axes.plot( time , mach , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('Mach Number',axis_font)
        axes.grid(True)     
        
        plt.savefig("Mission_envelope.png")
 

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

def mission_setup(analyses):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'mission1'
    
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
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    

    
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise1"
    
    segment.analyses.extend( analyses.base )
    
    segment.air_speed = 540
    segment.distance = 100*Units.km
    segment.altitude = 12 * Units.km

        
    mission.append_segment(segment)
    
#      

    
#    
     # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    

    
    segment = Segments.Cruise.Constant_Acceleration_Constant_Altitude(base_segment)
    segment.tag = "cruise2"
    
    segment.analyses.extend( analyses.base )
    
    segment.altitude = 12 * Units.km
    segment.air_speed_initial = 540*Units['m/s']
    segment.air_speed_final   = 690*Units['m/s']
    segment.acceleration = 0.7*Units['m/s^2']

        
    mission.append_segment(segment)
#    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    

    
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise_3"
    
    segment.analyses.extend( analyses.base )
    
    segment.air_speed = 690 * Units['m/s']
    segment.distance = 300*Units.km
    segment.altitude = 12* Units.km

        
    mission.append_segment(segment)
    
 
#    


    
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

    #plt.show()

