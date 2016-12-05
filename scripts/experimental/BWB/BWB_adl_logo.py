# full_setup.py
# 
# Created:  Aug 2014, E. Botero
# Modified: 





# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Core import (
Data, Container
)

from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Propulsion import compute_turbofan_geometry
from SUAVE.Plugins.OpenVSP.vsp_write import write

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():

    configs, analyses = full_setup()

    simple_sizing(configs)

    configs.finalize()
    analyses.finalize()

    # weight analysis
    #weights = analyses.configs.base.weights
    #breakdown = weights.evaluate()      

    ## mission analysis
    #mission = analyses.missions.base
    #results = mission.evaluate()

    ## plt the old results
    #plot_mission(results)

    return

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    # vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)
    
    # Write the vsp file
    write(vehicle, 'BWB')

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

    # adjust analyses for configs

    # takeoff_analysis
    analyses.takeoff.aerodynamics.settings.drag_coefficient_increment = 0.0000

    # landing analysis
    aerodynamics = analyses.landing.aerodynamics

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
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle

    aerodynamics.settings.drag_coefficient_increment = 0.0000
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

    # done!
    return analyses    

# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'BWB'    


    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff    = 79015.8   # kg
    vehicle.mass_properties.takeoff        = 79015.8   # kg
    vehicle.mass_properties.max_zero_fuel  = 0.9 * vehicle.mass_properties.max_takeoff
    vehicle.mass_properties.cargo          = 10000.  * Units.kilogram   

    # envelope properties
    vehicle.envelope.ultimate_load = 2.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 125.0     
    vehicle.passengers             = 170
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "medium range"


    # ------------------------------------------------------------------        
    #  Overall Body properties
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    #wing.aspect_ratio            = 5.86 #12. #not used in VSP definition
    wing.sweeps.quarter_chord    = 0. * Units.deg
    wing.thickness_to_chord      = 0.14
    wing.taper                   = 0.1
    wing.span_efficiency         = 0.9
    wing.dihedral                = 0.0 * Units.degrees

    wing.spans.projected         = 21#18.85 #38.7298

    wing.chords.root             = 10.#16.
    wing.chords.tip              = .1
    wing.chords.mean_aerodynamic = (2./3.)*(wing.chords.root + wing.chords.root -(wing.chords.root*wing.chords.root)/(wing.chords.root+wing.chords.root))

    wing.areas.reference         = 74.9

    wing.twists.root             = 0.#1.0 * Units.degrees
    wing.twists.tip              = 0. -4.0 * Units.degrees

    wing.origin                  = [3.,0.,0.]
    wing.aerodynamic_center      = [3.,0,0.]
    wing.percent_span_location   = 0.004085714
    wing.vertical                = False
    wing.symmetric               = True
    #wing.high_lift               = Fa

    wing.dynamic_pressure_ratio  = 1.0
    
    # Sections
    
    #nose
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_1'
    segment.percent_span_location = 0 
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 1.
    segment.dihedral_outboard     = 0. * Units.deg
    segment.sweeps.quarter_chord  = 0*Units.deg
    wing.Segments.append(segment)
    
    #first fuselage section
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_2'
    segment.percent_span_location = .004
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 1
    segment.dihedral_outboard     = 3. * Units.deg
    segment.sweeps.quarter_chord  = 50.*Units.deg
    
    #airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    #airfoil.coordinate_file = 'NACA2412'#'/Users/emiliobotero/Dropbox/SUAVE/Workspace/NACA2412.dat' # Or enter a NACA number
    
    #wing.append_airfoil(airfoil)
    #segment.append_airfoil(airfoil)
    
    wing.Segments.append(segment)
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_3'
    segment.percent_span_location = .164
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = .7 
    segment.dihedral_outboard     = 3. * Units.deg
    segment.sweeps.quarter_chord  = 30. * Units.deg
    
    #section = SUAVE.Components.Wings.Airfoils.Airfoil()
    #section.airfoil = 'my_airfoil.dat' # Or enter a NACA number
    
    #segment.append(section)
    wing.Segments.append(segment)    

    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_4'
    segment.percent_span_location = .284
    segment.twist                 = -1. * Units.deg
    segment.root_chord_percent    = 0.588
    segment.dihedral_outboard     = 3. * Units.deg
    segment.sweeps.quarter_chord  = 40. * Units.deg
    #section = SUAVE.Components.Wings.Airfoils.Airfoil()
    #section.airfoil = 'my_airfoil.dat' # Or enter a NACA number
    wing.Segments.append(segment)

    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_5'
    segment.percent_span_location = .378
    segment.twist                 = -1. * Units.deg
    segment.root_chord_percent    = 0.384
    segment.dihedral_outboard     = 3. * Units.deg
    segment.sweeps.quarter_chord  = 40. * Units.deg
    #section = SUAVE.Components.Wings.Airfoils.Airfoil()
    #section.airfoil = 'my_airfoil.dat' # Or enter a NACA number
    wing.Segments.append(segment)
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_6'
    segment.percent_span_location = .458
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.25
    segment.dihedral_outboard     = 6. * Units.deg
    segment.sweeps.quarter_chord  = 35. * Units.deg
    #section = SUAVE.Components.Wings.Airfoils.Airfoil()
    #section.airfoil = 'my_airfoil.dat' # Or enter a NACA number
    wing.Segments.append(segment)
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_7'
    segment.percent_span_location = .668
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.115
    segment.dihedral_outboard     = 6. * Units.deg
    segment.sweeps.quarter_chord  = 35. * Units.deg
    #section = SUAVE.Components.Wings.Airfoils.Airfoil()
    #section.airfoil = 'my_airfoil.dat' # Or enter a NACA number
    wing.Segments.append(segment)
    
    
    #wingtip
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_8'
    segment.percent_span_location = .877
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.1
    segment.dihedral_outboard     = 80. * Units.deg
    segment.sweeps.quarter_chord  = 45. * Units.deg
    #section = SUAVE.Components.Wings.Airfoils.Airfoil()
    #section.airfoil = 'my_airfoil.dat' # Or enter a NACA number
    wing.Segments.append(segment)
    
    # add to vehicle
    vehicle.append_component(wing)


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

    #computing the engine length and diameter
    compute_turbofan_geometry(turbofan,None)

    print "sls thrust : ",turbofan.sealevel_static_thrust
    print "engine length : ",turbofan.engine_length

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
    config.wings['main_wing'].slats.angle = 25. * Units.deg

    config.V2_VS_ratio = 1.21
    config.maximum_lift_coefficient = 2.

    configs.append(config)


    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------

    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'landing'

    config.wings['main_wing'].flaps_angle = 30. * Units.deg
    config.wings['main_wing'].slats_angle = 25. * Units.deg

    config.Vref_VS_ratio = 1.23
    config.maximum_lift_coefficient = 2.

    configs.append(config)


    # done!
    return configs

# ----------------------------------------------------------------------
#   Sizing for the Vehicle Configs
# ----------------------------------------------------------------------

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
        Lift   = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0] / Units.lbf
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0] / Units.lbf
        eta  = segment.conditions.propulsion.throttle[:,0]
        mdot   = segment.conditions.weights.vehicle_mass_rate[:,0]
        thrust =  segment.conditions.frames.body.thrust_force_vector[:,0]
        sfc    = 3600. * mdot / 0.1019715 / thrust	


        axes = fig.add_subplot(2,1,1)
        axes.plot( time , Thrust , line_style )
        axes.set_ylabel('Thrust (lbf)',axis_font)
        axes.grid(True)

        axes = fig.add_subplot(2,1,2)
        axes.plot( time , eta , line_style )
        axes.set_xlabel('Time (min)',axis_font)
        axes.set_ylabel('eta',axis_font)
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
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]
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

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        CLift  = segment.conditions.aerodynamics.lift_coefficient[:,0]
        CDrag  = segment.conditions.aerodynamics.drag_coefficient[:,0]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]
        aoa    = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        l_d    = CLift/CDrag
        mass   = segment.conditions.weights.total_mass[:,0] / Units.lb
        altitude = segment.conditions.freestream.altitude[:,0] / Units.ft
        mdot   = segment.conditions.weights.vehicle_mass_rate[:,0]
        thrust =  segment.conditions.frames.body.thrust_force_vector[:,0]
        sfc    = 3600. * mdot / 0.1019715 / thrust	

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
    #   Aerodynamics 2
    # ------------------------------------------------------------------
    fig = plt.figure("Velocities",figsize=(8,10))
    for segment in results.segments.values():

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Lift   = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0] / Units.lbf
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0] / Units.lbf
        eta  = segment.conditions.propulsion.throttle[:,0]
        mdot   = segment.conditions.weights.vehicle_mass_rate[:,0]
        thrust =  segment.conditions.frames.body.thrust_force_vector[:,0]
        sfc    = 3600. * mdot / 0.1019715 / thrust
        velocity  = segment.conditions.freestream.velocity[:,0]
        pressure  = segment.conditions.freestream.pressure[:,0]
        density  = segment.conditions.freestream.density[:,0]
        EAS = velocity * np.sqrt(density/1.225)
        mach = segment.conditions.freestream.mach_number[:,0]


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
    #base.fuselages['fuselage'].number_coach_seats = base.passengers

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
    base_segment.process.iterate.conditions.stability     = SUAVE.Methods.skip
    base_segment.process.finalize.post_process.stability  = SUAVE.Methods.skip


    # ------------------------------------------------------------------
    #   First Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"

    segment.analyses.extend( analyses.takeoff )

    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 3.0   * Units.km
    segment.air_speed      = 125.0 * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']

    # add to misison
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Second Climb Segment: constant Speed, constant segment angle 
    # ------------------------------------------------------------------    

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_2"

    segment.analyses.extend( analyses.cruise )

    segment.altitude_end   = 8.0   * Units.km
    segment.air_speed      = 190.0 * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Third Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_3"

    segment.analyses.extend( analyses.cruise )

    segment.altitude_end = 10.668 * Units.km
    segment.air_speed    = 226.0  * Units['m/s']
    segment.climb_rate   = 3.0    * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend( analyses.cruise )

    segment.air_speed  = 230.412 * Units['m/s']
    segment.distance   = (3933.65 + 770 - 92.6) * Units.km

    # add to mission
    mission.append_segment(segment)


# ------------------------------------------------------------------
#   First Descent Segment: consant speed, constant segment rate
# ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_1"

    segment.analyses.extend( analyses.cruise )

    segment.altitude_end = 8.0   * Units.km
    segment.air_speed    = 220.0 * Units['m/s']
    segment.descent_rate = 4.5   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_2"

    segment.analyses.extend( analyses.landing )

    analyses.landing.aerodynamics.settings.spoiler_drag_increment = 0.00

    segment.altitude_end = 6.0   * Units.km
    segment.air_speed    = 195.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Third Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_3"

    segment.analyses.extend( analyses.landing )

    analyses.landing.aerodynamics.settings.spoiler_drag_increment = 0.00

    segment.altitude_end = 4.0   * Units.km
    segment.air_speed    = 170.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Fourth Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_4"

    segment.analyses.extend( analyses.landing )

    analyses.landing.aerodynamics.settings.spoiler_drag_increment = 0.00

    segment.altitude_end = 2.0   * Units.km
    segment.air_speed    = 150.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']


    # add to mission
    mission.append_segment(segment)



    # ------------------------------------------------------------------
    #   Fifth Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_5"

    segment.analyses.extend( analyses.landing )
    analyses.landing.aerodynamics.settings.spoiler_drag_increment = 0.00


    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 145.0 * Units['m/s']
    segment.descent_rate = 3.0   * Units['m/s']


    # append to mission
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


    # ------------------------------------------------------------------
    #   Mission for Constrained Fuel
    # ------------------------------------------------------------------    
    fuel_mission = SUAVE.Analyses.Mission.Mission() #Fuel_Constrained()
    fuel_mission.tag = 'fuel'
    fuel_mission.range   = 1277. * Units.nautical_mile
    fuel_mission.payload   = 19000.
    missions.append(fuel_mission)    


    # ------------------------------------------------------------------
    #   Mission for Constrained Short Field
    # ------------------------------------------------------------------    
    short_field = SUAVE.Analyses.Mission.Mission(base_mission) #Short_Field_Constrained()
    short_field.tag = 'short_field'    

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    airport.available_tofl = 1500.
    short_field.airport = airport    
    missions.append(short_field)



    # ------------------------------------------------------------------
    #   Mission for Fixed Payload
    # ------------------------------------------------------------------    
    payload = SUAVE.Analyses.Mission.Mission() #Payload_Constrained()
    payload.tag = 'payload'
    payload.range   = 2316. * Units.nautical_mile
    payload.payload   = 19000.
    missions.append(payload)


    # done!
    return missions  

if __name__ == '__main__': 
    main()    
    plt.show()

