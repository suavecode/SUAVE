# tut_mission_B737.py
# 
# Created:  Aug 2014, SUAVE Team
# Modified: Jun 2015, SUAVE Team

""" setup file for a mission with a 737
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
Data, Container
)


from SUAVE.Plugins.OpenVSP import write

from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Propulsion import compute_turbofan_geometry

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
    weights = SUAVE.Analyses.Weights.Weights()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.SU2_Euler()
    #aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle

    aerodynamics.process.compute.lift.inviscid.settings.parallel   = True
    aerodynamics.process.compute.lift.inviscid.settings.processors = 12
     
    aerodynamics.process.compute.lift.inviscid.training.angle_of_attack  = np.array([-2.,3.,8.,12.]) * Units.deg
    #aerodynamics.process.compute.lift.inviscid.training_file       = 'base_data.txt'
    
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
    vehicle.tag = 'Boeing_737_800'    


    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff               = 79015.8   # kg
    vehicle.mass_properties.takeoff                   = 79015.8   # kg
    vehicle.mass_properties.max_zero_fuel             = 0.9 * vehicle.mass_properties.max_takeoff
    vehicle.mass_properties.cargo                     = 10000.  * Units.kilogram   

    # envelope properties
    vehicle.envelope.ultimate_load = 2.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 124.862       
    vehicle.passengers             = 170
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "medium range"


    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.aspect_ratio            = 10.18 # Not set
    wing.thickness_to_chord      = 0.1 # Not set
    wing.taper                   = 0.782/7.7760
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 34.32    

    wing.chords.root             = 7.760 * Units.meter
    wing.chords.tip              = 0.782 * Units.meter
    wing.chords.mean_aerodynamic = 4.235 * Units.meter

    wing.areas.reference         = 124.862  # Not set
    wing.sweeps.quarter_chord    = 25. * Units.degrees

    wing.twists.root             = 4.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    wing.dihedral                = 2.5 * Units.degrees

    wing.origin                  = [13.61,0,-1.27]
    wing.aerodynamic_center      = [0,0,0] 

    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True

    wing.dynamic_pressure_ratio  = 1.0
    
    airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    airfoil.coordinate_file = 'B737b.dat'
    wing.append_airfoil(airfoil)    
    
    # Root
    root_airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    root_airfoil.coordinate_file = 'B737a.dat'
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'root'
    segment.percent_span_location = 0.0
    segment.twist                 = 4. * Units.deg
    segment.root_chord_percent    = 1.
    segment.dihedral_outboard     = 2.5 * Units.degrees
    segment.sweeps.quarter_chord  = 28.225 * Units.degrees
    segment.append_airfoil(root_airfoil)
    wing.Segments.append(segment)    
    
    
    # Yehudi
    yehudi_airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    yehudi_airfoil.coordinate_file = 'B737b.dat'
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'yehudi'
    segment.percent_span_location = 0.324
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.5
    segment.dihedral_outboard     = 5.5 * Units.degrees
    segment.sweeps.quarter_chord  = 25. * Units.degrees
    segment.append_airfoil(yehudi_airfoil)
    wing.Segments.append(segment)
    
    # set tip section start point
    mid_airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    mid_airfoil.coordinate_file = 'B737c.dat' 
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_2'
    segment.percent_span_location = 0.963
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.220
    segment.dihedral_outboard     = 5.5 * Units.degrees
    segment.sweeps.quarter_chord  = 56.75 * Units.degrees
    segment.append_airfoil(mid_airfoil)
    wing.Segments.append(segment)
    
    # Add a tip
    tip_airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    tip_airfoil.coordinate_file = 'B737c.dat' 
    
    segment = SUAVE.Components.Wings.Segment() 
    segment.tag                   = 'Tip'
    segment.percent_span_location = 1.
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.782/7.760
    segment.dihedral_outboard     = 0.
    segment.sweeps.quarter_chord  = 0.
    segment.append_airfoil(tip_airfoil)
    wing.Segments.append(segment)        
    


    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'

    wing.aspect_ratio            = 6.16 # Not set
    wing.thickness_to_chord      = 0.08 # Not set
    wing.taper                   = 0.955/4.70
    wing.span_efficiency         = 0.9

    wing.spans.projected         = 14.2

    wing.chords.root             = 4.70
    wing.chords.tip              = 0.955   
    wing.chords.mean_aerodynamic = 2.5

    wing.areas.reference         = 42.65

    wing.twists.root             = 0.0 * Units.degrees # Not set
    wing.twists.tip              = 0.0 * Units.degrees # Not set
    
    wing.sweeps.quarter_chord    = 40.0 * Units.degrees

    wing.origin                  = [32.83,0,1.14]
    wing.aerodynamic_center      = [0,0,0]
    wing.dihedral                = 8.63 * Units.degrees

    wing.vertical                = False 
    wing.symmetric               = True

    wing.dynamic_pressure_ratio  = 0.9 
    
    segment = SUAVE.Components.Wings.Segment() 
    segment.tag                   = 'root_segment'
    segment.percent_span_location = 0.0
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 1.0
    segment.dihedral_outboard     = 8.63 * Units.degrees
    segment.sweeps.quarter_chord  = 38.42 * Units.degrees
    wing.Segments.append(segment)         
    
    
    segment = SUAVE.Components.Wings.Segment() 
    segment.tag                   = 'segment'
    segment.percent_span_location = 0.91
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.35
    segment.dihedral_outboard     = 8.63 * Units.degrees
    segment.sweeps.quarter_chord  = 48.17 * Units.degrees
    wing.Segments.append(segment)         

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'    

    wing.aspect_ratio            = 1.91
    wing.sweep                   = 25. * Units.deg
    wing.thickness_to_chord      = 0.08
    wing.taper                   = 0.25
    wing.span_efficiency         = 0.9

    wing.spans.projected         = 7.77

    wing.chords.root             = 8.19
    wing.chords.tip              = 0.95
    wing.chords.mean_aerodynamic = 4.0

    wing.areas.reference         = 27.316

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  

    wing.origin                  = [28.79,0,1.54]
    wing.aerodynamic_center      = [0,0,0]    

    wing.vertical                = True 
    wing.symmetric               = False
    wing.t_tail                  = False
    
    segment = SUAVE.Components.Wings.Segment() 
    segment.tag                   = 'root'
    segment.percent_span_location = 0.0
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 1.
    segment.dihedral_outboard     = 0 * Units.degrees
    segment.sweeps.quarter_chord  = 63.63 * Units.degrees
    wing.Segments.append(segment)      
    
    segment = SUAVE.Components.Wings.Segment() 
    segment.tag                   = 'segment_1'
    segment.percent_span_location = 0.194
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.540
    segment.dihedral_outboard     = 0. * Units.degrees
    segment.sweeps.quarter_chord  = 30.0 * Units.degrees
    wing.Segments.append(segment)  
    
    segment = SUAVE.Components.Wings.Segment() 
    segment.tag                   = 'segment_2'
    segment.percent_span_location = 0.961
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.175
    segment.dihedral_outboard     = 0.0 * Units.degrees
    segment.sweeps.quarter_chord  = 51.0 * Units.degrees
    wing.Segments.append(segment)        

    wing.dynamic_pressure_ratio  = 1.0

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------

    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'

    fuselage.seats_abreast         = 6
    fuselage.seat_pitch            = 1

    fuselage.fineness.nose         = 1.57
    fuselage.fineness.tail         = 3.2

    fuselage.lengths.nose          = 8.0
    fuselage.lengths.tail          = 12.
    fuselage.lengths.cabin         = 28.85
    fuselage.lengths.total         = 38.02
    fuselage.lengths.fore_space    = 6.
    fuselage.lengths.aft_space     = 5.    

    fuselage.width                 = 3.76

    fuselage.heights.maximum       = 3.76
    fuselage.heights.at_quarter_length          = 3.76
    fuselage.heights.at_three_quarters_length   = 3.65
    fuselage.heights.at_wing_root_quarter_chord = 3.76

    fuselage.areas.side_projected  = 142.1948
    fuselage.areas.wetted          = 446.718
    fuselage.areas.front_projected = 12.57

    fuselage.effective_diameter    = 3.76

    fuselage.differential_pressure = 5.0e4 * Units.pascal # Maximum differential pressure
    
    fuselage.OpenVSP_values = Data() # VSP uses degrees directly
    
    fuselage.OpenVSP_values.nose = Data()
    fuselage.OpenVSP_values.nose.top = Data()
    fuselage.OpenVSP_values.nose.side = Data()
    fuselage.OpenVSP_values.nose.top.angle = 75.0
    fuselage.OpenVSP_values.nose.top.strength = 0.40
    fuselage.OpenVSP_values.nose.side.angle = 45.0
    fuselage.OpenVSP_values.nose.side.strength = 0.75  
    fuselage.OpenVSP_values.nose.TB_Sym = True
    fuselage.OpenVSP_values.nose.z_pos = -.015
    
    fuselage.OpenVSP_values.tail = Data()
    fuselage.OpenVSP_values.tail.top = Data()
    fuselage.OpenVSP_values.tail.side = Data()    
    fuselage.OpenVSP_values.tail.bottom = Data()
    fuselage.OpenVSP_values.tail.top.angle = -90.0
    fuselage.OpenVSP_values.tail.top.strength = 0.1
    fuselage.OpenVSP_values.tail.side.angle = -30.0
    fuselage.OpenVSP_values.tail.side.strength = 0.50  
    fuselage.OpenVSP_values.tail.TB_Sym = True
    fuselage.OpenVSP_values.tail.bottom.angle = -30.0
    fuselage.OpenVSP_values.tail.bottom.strength = 0.50 
    fuselage.OpenVSP_values.tail.z_pos = .035

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
    turbofan.engine_length     = 4.1
    turbofan.nacelle_diameter  = 0.85
    turbofan.origin            = [[13.72, 4.86,-1.9],[13.72, -4.86,-1.9]]

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
    
    #write(vehicle,base_config.tag) 


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
    #   First Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"

    segment.analyses.extend( analyses.base )

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

    segment.analyses.extend( analyses.base )

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

    segment.analyses.extend( analyses.base )

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

    segment.analyses.extend( analyses.base )

    segment.air_speed  = 230.412 * Units['m/s']
    segment.distance   = (3933.65 + 770 - 92.6) * Units.km

    # add to mission
    mission.append_segment(segment)


# ------------------------------------------------------------------
#   First Descent Segment: consant speed, constant segment rate
# ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_1"

    segment.analyses.extend( analyses.base )

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

    segment.analyses.extend( analyses.base )

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

    segment.analyses.extend( analyses.base )

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

    segment.analyses.extend( analyses.base )

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

    segment.analyses.extend( analyses.base )

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

