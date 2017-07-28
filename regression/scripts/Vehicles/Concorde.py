# Concorde.py
#
# Created:  Feb 2017, M. Vegh (created from data taken from concorde/concorde.py)
# Modified: Jul 2017, T. MacDonald

""" setup file for the Concorde 
"""

import numpy as np
import SUAVE
from SUAVE.Core import Units
from SUAVE.Core import (
    Data, Container,
)
from SUAVE.Methods.Propulsion.turbojet_sizing import turbojet_sizing


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
    vehicle.mass_properties.max_takeoff               = 185000.   # kg
    vehicle.mass_properties.operating_empty           = 78700.   # kg
    vehicle.mass_properties.takeoff                   = 185000.   # kg
    vehicle.mass_properties.cargo                     = 1000.  * Units.kilogram   
        
    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 358.25      
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
    
    wing.spans.projected         = 25.6    
    
    wing.chords.root             = 33.8
    wing.total_length            = 33.8
    wing.chords.tip              = 1.1
    wing.chords.mean_aerodynamic = 18.4
    
    wing.areas.reference         = 358.25 
    wing.areas.wetted            = 653. - 12.*2.4*2 # 2.4 is engine area on one side
    wing.areas.exposed           = 326.5
    wing.areas.affected          = .6*wing.areas.reference
    
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
    
    wing_airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    wing_airfoil.coordinate_file = 'NACA65-203.dat' 
    
    wing.append_airfoil(wing_airfoil)  
    
    # set root sweep with inner section
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_1'
    segment.percent_span_location = 0.
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 33.8/33.8
    segment.dihedral_outboard     = 0.
    segment.sweeps.quarter_chord  = 67. * Units.deg
    segment.vsp_mesh              = Data()
    segment.vsp_mesh.inner_radius    = 1./source_ratio
    segment.vsp_mesh.outer_radius    = 1./source_ratio
    segment.vsp_mesh.inner_length    = .044/source_ratio
    segment.vsp_mesh.outer_length    = .044/source_ratio
    segment.vsp_mesh.matching_TE     = False
    segment.append_airfoil(wing_airfoil)
    wing.Segments.append(segment)
    
    # set mid section start point
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_2'
    segment.percent_span_location = 6.15/(25.6/2) + wing.Segments['section_1'].percent_span_location
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 13.8/33.8
    segment.dihedral_outboard     = 0.
    segment.sweeps.quarter_chord  = 48. * Units.deg
    segment.vsp_mesh              = Data()
    segment.vsp_mesh.inner_radius    = 1./source_ratio
    segment.vsp_mesh.outer_radius    = .88/source_ratio
    segment.vsp_mesh.inner_length    = .044/source_ratio
    segment.vsp_mesh.outer_length    = .044/source_ratio 
    segment.vsp_mesh.matching_TE     = False
    segment.append_airfoil(wing_airfoil)
    wing.Segments.append(segment)
    
    # set tip section start point
    segment = SUAVE.Components.Wings.Segment() 
    segment.tag                   = 'section_3'
    segment.percent_span_location = 5.95/(25.6/2) + wing.Segments['section_2'].percent_span_location
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 4.4/33.8
    segment.dihedral_outboard     = 0.
    segment.sweeps.quarter_chord  = 71. * Units.deg 
    segment.vsp_mesh              = Data()
    segment.vsp_mesh.inner_radius    = .88/source_ratio
    segment.vsp_mesh.outer_radius    = .22/source_ratio
    segment.vsp_mesh.inner_length    = .044/source_ratio
    segment.vsp_mesh.outer_length    = .011/source_ratio 
    segment.append_airfoil(wing_airfoil)
    wing.Segments.append(segment)    
    
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
    wing.areas.wetted            = 76. 
    wing.areas.exposed           = 38.
    wing.areas.affected          = 33.91
    
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  
    
    wing.origin                  = [42.,0,1.]
    wing.aerodynamic_center      = [50,0,0]    
    
    wing.vertical                = True 
    wing.symmetric               = False
    wing.t_tail                  = False
    wing.high_mach               = True     
    
    wing.dynamic_pressure_ratio  = 1.0
    
    tail_airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    tail_airfoil.coordinate_file = 'supertail_refined.dat' 
    
    wing.append_airfoil(tail_airfoil)  

    # set root sweep with inner section
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_1'
    segment.percent_span_location = 0.0
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 14.5/14.5
    segment.dihedral_outboard     = 0.
    segment.sweeps.quarter_chord  = 63. * Units.deg
    segment.vsp_mesh              = Data()
    segment.vsp_mesh.inner_radius    = 2.9/source_ratio
    segment.vsp_mesh.outer_radius    = 1.5/source_ratio
    segment.vsp_mesh.inner_length    = .044/source_ratio
    segment.vsp_mesh.outer_length    = .044/source_ratio
    segment.append_airfoil(tail_airfoil)
    wing.Segments.append(segment)
    
    # set mid section start point
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_2'
    segment.percent_span_location = 2.4/(6.0) + wing.Segments['section_1'].percent_span_location
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 7.5/14.5
    segment.dihedral_outboard     = 0.
    segment.sweeps.quarter_chord  = 40. * Units.deg
    segment.vsp_mesh              = Data()
    segment.vsp_mesh.inner_radius    = 1.5/source_ratio
    segment.vsp_mesh.outer_radius    = .54/source_ratio
    segment.vsp_mesh.inner_length    = .044/source_ratio
    segment.vsp_mesh.outer_length    = .027/source_ratio 
    segment.append_airfoil(tail_airfoil)
    wing.Segments.append(segment)
    
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

    fuselage.areas.wetted          = 447.
    fuselage.areas.front_projected = 11.9
    
    
    fuselage.effective_diameter    = 3.1
    
    fuselage.differential_pressure = 7.4e4 * Units.pascal    # Maximum differential pressure
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
        
    # ------------------------------------------------------------------
    #   Turbojet Network
    # ------------------------------------------------------------------    
    
    # instantiate the gas turbine network
    turbojet = SUAVE.Components.Energy.Networks.Turbojet_Super()
    turbojet.tag = 'turbojet'
    
    # setup
    turbojet.number_of_engines = 4.0
    turbojet.engine_length     = 12.0
    turbojet.nacelle_diameter  = 1.3
    turbojet.inlet_diameter    = 1.1
    turbojet.areas             = Data()
    turbojet.areas.wetted      = 12.5*4.7*2. # 4.7 is outer perimeter on one side
    turbojet.origin            = [[37.,6.,-1.3],[37.,5.3,-1.3],[37.,-5.3,-1.3],[37.,-6.,-1.3]]
    
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
    
    # instantiate
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag = 'inlet_nozzle'
    
    # setup
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio        = 1.0
    
    # add to network
    turbojet.append(inlet_nozzle)
    
    
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
    combustor.alphac                    = 1.0     
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
    #  Component 9 - Divergening Nozzle
    
    
    
    
    # ------------------------------------------------------------------
    #Component 10 : thrust (to compute the thrust)
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='compute_thrust'
 
    #total design thrust (includes all the engines)
    thrust.total_design             = 4*140000. * Units.N #Newtons
 
    # Note: Sizing builds the propulsor. It does not actually set the size of the turbojet
    #design sizing conditions
    altitude      = 0.0*Units.ft
    mach_number   = 0.01
    isa_deviation = 0.
    
    # add to network
    turbojet.thrust = thrust

    #size the turbojet
    turbojet_sizing(turbojet,mach_number,altitude)   
    
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
    
    # done!
    return configs