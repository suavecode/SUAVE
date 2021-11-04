# Boeing_BWB_450.py
#
# Created:  Feb 2017, M. Vegh (created from data originally SU2_surrogate/BWB-450.py)
# Modified: 

""" setup file for the BWB vehicle
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
from SUAVE.Core import Units
from SUAVE.Core import Data, Container
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Propulsion import compute_turbofan_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import segment_properties

from copy import deepcopy

# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------


def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Boeing_BWB_450'    

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # mass properties
    vehicle.mass_properties.max_takeoff               = 823000. * Units.lb
    vehicle.mass_properties.takeoff                   = 823000. * Units.lb
    vehicle.mass_properties.max_zero_fuel             = 0.9 * vehicle.mass_properties.max_takeoff
    vehicle.mass_properties.cargo                     = 00.  * Units.kilogram   

    # envelope properties
    vehicle.envelope.ultimate_load = 2.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 7840. * 2 * Units.feet**2       
    vehicle.passengers             = 450.
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "medium range"


    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.aspect_ratio            = 289.**2 / (7840. * 2)
    wing.thickness_to_chord      = 0.15
    wing.taper                   = 0.0138
    wing.spans.projected         = 289.0 * Units.feet  
    wing.chords.root             = 145.0 * Units.feet
    wing.chords.tip              = 3.5  * Units.feet
    wing.chords.mean_aerodynamic = 80. * Units.feet
    wing.areas.reference         = 7840. * 2 * Units.feet**2
    wing.sweeps.quarter_chord    = 33. * Units.degrees
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    wing.dihedral                = 2.5 * Units.degrees
    wing.origin                  = [[0.,0.,0]]
    wing.aerodynamic_center      = [0,0,0] 
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True
    wing.dynamic_pressure_ratio  = 1.0

    segment = SUAVE.Components.Wings.Segment()

    segment.tag                   = 'section_1'
    segment.percent_span_location = 0.0
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 1.
    segment.dihedral_outboard     = 0. * Units.degrees
    segment.sweeps.quarter_chord  = 40.0 * Units.degrees
    segment.thickness_to_chord    = 0.165
    segment.vsp_mesh              = Data()
    segment.vsp_mesh.inner_radius    = 4.
    segment.vsp_mesh.outer_radius    = 4.
    segment.vsp_mesh.inner_length    = .14
    segment.vsp_mesh.outer_length    = .14    
    wing.Segments.append(segment)    
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                      = 'section_2'
    segment.percent_span_location    = 0.052
    segment.twist                    = 0. * Units.deg
    segment.root_chord_percent       = 0.921
    segment.dihedral_outboard        = 0.   * Units.degrees
    segment.sweeps.quarter_chord     = 52.5 * Units.degrees
    segment.thickness_to_chord       = 0.167
    segment.vsp_mesh                 = Data()
    segment.vsp_mesh.inner_radius    = 4.
    segment.vsp_mesh.outer_radius    = 4.
    segment.vsp_mesh.inner_length    = .14
    segment.vsp_mesh.outer_length    = .14     
    wing.Segments.append(segment)   

    segment = SUAVE.Components.Wings.Segment()
    segment.tag                      = 'section_3'
    segment.percent_span_location    = 0.138
    segment.twist                    = 0. * Units.deg
    segment.root_chord_percent       = 0.76
    segment.dihedral_outboard        = 1.85 * Units.degrees
    segment.sweeps.quarter_chord     = 36.9 * Units.degrees  
    segment.thickness_to_chord       = 0.171
    segment.vsp_mesh                 = Data()
    segment.vsp_mesh.inner_radius    = 4.
    segment.vsp_mesh.outer_radius    = 4.
    segment.vsp_mesh.inner_length    = .14
    segment.vsp_mesh.outer_length    = .14     
    wing.Segments.append(segment)   
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                      = 'section_4'
    segment.percent_span_location    = 0.221
    segment.twist                    = 0. * Units.deg
    segment.root_chord_percent       = 0.624
    segment.dihedral_outboard        = 1.85 * Units.degrees
    segment.sweeps.quarter_chord     = 30.4 * Units.degrees    
    segment.thickness_to_chord       = 0.175
    segment.vsp_mesh                 = Data()
    segment.vsp_mesh.inner_radius    = 4.
    segment.vsp_mesh.outer_radius    = 2.8
    segment.vsp_mesh.inner_length    = .14
    segment.vsp_mesh.outer_length    = .14     
    wing.Segments.append(segment)       
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_5'
    segment.percent_span_location = 0.457
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.313
    segment.dihedral_outboard     = 1.85  * Units.degrees
    segment.sweeps.quarter_chord  = 30.85 * Units.degrees
    segment.thickness_to_chord    = 0.118
    wing.Segments.append(segment)       
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_6'
    segment.percent_span_location = 0.568
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.197
    segment.dihedral_outboard     = 1.85 * Units.degrees
    segment.sweeps.quarter_chord  = 34.3 * Units.degrees
    segment.thickness_to_chord    = 0.10
    wing.Segments.append(segment)     
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_7'
    segment.percent_span_location = 0.97
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.086
    segment.dihedral_outboard     = 73. * Units.degrees
    segment.sweeps.quarter_chord  = 55. * Units.degrees
    segment.thickness_to_chord    = 0.10
    wing.Segments.append(segment)      

    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'tip'
    segment.percent_span_location = 1
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.0241
    segment.dihedral_outboard     = 0. * Units.degrees
    segment.sweeps.quarter_chord  = 0. * Units.degrees
    segment.thickness_to_chord    = 0.10
    wing.Segments.append(segment)  
    
    # Fill out more segment properties automatically
    wing = segment_properties(wing)        

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #   Nacelle  
    # ------------------------------------------------------------------
    nacelle                       = SUAVE.Components.Nacelles.Nacelle()
    nacelle.diameter              = 3.96 * Units.meters 
    nacelle.length                = 289. * Units.inches
    nacelle.tag                   = 'nacelle' 
    nacelle.origin                = [[133.0 *Units.feet, 25.0*Units.feet, 6.5*Units.feet]]
    nacelle.areas.wetted          =  nacelle.length *(2*np.pi*nacelle.diameter/2.) 

    nacelle_2                     = deepcopy(nacelle)
    nacelle_2.tag                 = 'nacelle_2' 
    nacelle_2.origin              = [[145.0 *Units.feet, 0.0*Units.feet, 6.5*Units.feet]]     
     
    nacelle_3                     = deepcopy(nacelle)
    nacelle_3.tag                 = 'nacelle_3'
    nacelle_3.origin              = [[133.0 *Units.feet, -25.0*Units.feet, 6.5*Units.feet]]   
    
    vehicle.append_component(nacelle) 
    vehicle.append_component(nacelle_2) 
    vehicle.append_component(nacelle_3) 
    
    # ------------------------------------------------------------------
    #   Turbofan Network
    # ------------------------------------------------------------------
    #instantiate the gas turbine network
    turbofan     = SUAVE.Components.Energy.Networks.Turbofan()
    turbofan.tag = 'turbofan1'

    # setup
    turbofan.number_of_engines = 3.0
    turbofan.bypass_ratio      = 8.1
    turbofan.engine_length     = 289. * Units.inches 
    turbofan.origin            = [[133.0 *Units.feet, 25.0*Units.feet, 6.5*Units.feet],[145.0 *Units.feet, 0.0*Units.feet, 6.5*Units.feet],[133.0 *Units.feet, -25.0*Units.feet, 6.5*Units.feet]]
    
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
    inlet_nozzle.polytropic_efficiency = 1.0
    inlet_nozzle.pressure_ratio        = 1.0
    # add to network
    turbofan.append(inlet_nozzle)
    
    # ------------------------------------------------------------------
    #  Component 3 - Low Pressure Compressor
    
    # instantiate
    compressor = SUAVE.Components.Energy.Converters.Compressor()
    compressor.tag = 'low_pressure_compressor'
    # setup
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio        = 1.1
    # add to network
    turbofan.append(compressor)
    
    # ------------------------------------------------------------------
    #  Component 4 - High Pressure Compressor
    
    # instantiate
    compressor = SUAVE.Components.Energy.Converters.Compressor()
    compressor.tag = 'high_pressure_compressor'
    # setup
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio        = 23.0
    #compressor.hub_to_tip_ratio      = 0.325
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
    combustor.efficiency                = 1.0
    combustor.alphac                    = 1.0
    combustor.turbine_inlet_temperature = 1592. * Units.kelvin
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
    fan.pressure_ratio        = 1.58
    # add to network
    turbofan.append(fan)
    
    # ------------------------------------------------------------------
    #Component 10 : thrust (to compute the thrust)
    thrust = SUAVE.Components.Energy.Processes.Thrust()
    thrust.tag ='compute_thrust'
    
    #total design thrust (includes all the engines)
    thrust.total_design  = 2.0*512000 * Units.N
    thrust.bypass_ratio  = 8.4
    
    #design sizing conditions
    altitude      = 0. * Units.km
    mach_number   = 0.01
    isa_deviation = 0.
    
    # add to network
    turbofan.thrust = thrust
    
    #size the turbofan
    turbofan_sizing(turbofan,mach_number,altitude)
    #turbofan.size(mach_number,altitude)
    
    #computing the engine length and diameter
    for nac in vehicle.nacelles: 
        compute_turbofan_geometry(turbofan,nac)
    
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
