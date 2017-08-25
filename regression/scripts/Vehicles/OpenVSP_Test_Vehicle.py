# OpenVSP_Test_Vehicle.py
#
# Created:  Feb 2017, M. Vegh (taken from data originally in B737/mission_B737.py, noise_optimization/Vehicles.py, and Boeing 737 tutorial script
# Modified: Jul 2017, M. Clarke

""" setup file for the Boeing 737 vehicle
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Methods.Propulsion.turbojet_sizing import turbojet_sizing
from SUAVE.Input_Output.OpenVSP.vsp_write import write


# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'VSP_vehicle'    
    
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff               = 79015.8   # kg
    vehicle.mass_properties.takeoff                   = 79015.8   # kg
    vehicle.mass_properties.operating_empty           = 62746.4   # kg
    vehicle.mass_properties.max_zero_fuel             = 62732.0   # kg
    
 
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
    
    wing.aspect_ratio            = 10.18
    wing.sweeps.quarter_chord    = 25 * Units.deg
    wing.thickness_to_chord      = 0.1
    wing.taper                   = 0.1
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 34.32   
    
    wing.chords.root             = 7.760 * Units.meter
    wing.chords.tip              = 0.782 * Units.meter
    wing.chords.mean_aerodynamic = 4.235 * Units.meter
    wing.total_length            = 10.
    
    wing.areas.reference         = 124.862 
    
    wing.twists.root             = 4.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    
    wing.origin                  = [13.61,1.,-.5]
    wing.aerodynamic_center      = [0,0,0]
    
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True
    
    wing.dynamic_pressure_ratio  = 1.0
    
    wing_airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    wing_airfoil.coordinate_file = 'NACA65-203.dat'     
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_1'
    segment.percent_span_location = 0.
    segment.twist                 = 7.5 * Units.deg
    segment.root_chord_percent    = 6.2/6.2
    segment.dihedral_outboard     = 0.
    segment.sweeps.quarter_chord  = 2.5 * Units.deg
    segment_airfoil               = SUAVE.Components.Wings.Airfoils.Airfoil()
    segment_airfoil.coordinate_file  = 'sc20406.dat' 
    segment.append_airfoil(segment_airfoil)
    wing.Segments.append(segment)    
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_2'
    segment.percent_span_location = 2.6/5.2
    segment.root_chord_percent    = 2.3/6.2
    segment.dihedral_outboard     = 0.
    segment.sweeps.quarter_chord  = 45. * Units.deg
    segment_airfoil               = SUAVE.Components.Wings.Airfoils.Airfoil()
    segment_airfoil.coordinate_file  = 'sc20414.dat'     
    segment.append_airfoil(segment_airfoil)
    wing.Segments.append(segment)       
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_3'
    segment.percent_span_location = 4.6/5.2
    segment.twist                 = 5. * Units.deg
    segment.root_chord_percent    = 2.3/6.2
    segment.dihedral_outboard     = 0.
    segment.sweeps.quarter_chord  = 65. * Units.deg
    segment_airfoil               = SUAVE.Components.Wings.Airfoils.Airfoil()
    segment_airfoil.coordinate_file  = 'sc20414.dat'     
    segment.append_airfoil(segment_airfoil)
    wing.Segments.append(segment)         
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_4'
    segment.percent_span_location = 1.
    segment.twist                 = -5. * Units.deg
    segment.root_chord_percent    = 2.3/6.2
    segment.dihedral_outboard     = 0.
    segment.sweeps.quarter_chord  = 65. * Units.deg
    segment_airfoil               = SUAVE.Components.Wings.Airfoils.Airfoil()
    segment_airfoil.coordinate_file  = 'sc20406.dat'     
    segment.append_airfoil(segment_airfoil)
    wing.Segments.append(segment)          

    vehicle.append_component(wing)

    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'
    
    wing.aspect_ratio            = 10.18
    wing.sweeps.quarter_chord    = 25 * Units.deg
    wing.thickness_to_chord      = 0.1
    wing.taper                   = 0.1
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 34.32   
    
    wing.chords.root             = 7.760 * Units.meter
    wing.chords.tip              = 0.782 * Units.meter
    wing.chords.mean_aerodynamic = 4.235 * Units.meter
    wing.total_length            = 10.
    
    wing.areas.reference         = 124.862 
    
    wing.twists.root             = 4.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    
    wing.origin                  = [26.,0,0.2]
    wing.aerodynamic_center      = [0,0,0]
    
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True
    
    wing.dynamic_pressure_ratio  = 1.0
    
    wing_airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    wing_airfoil.coordinate_file = 'NACA65-203.dat'     
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_1'
    segment.percent_span_location = 2.6/5.2
    segment.twist                 = -10. * Units.deg
    segment.root_chord_percent    = 2.3/6.2
    segment.dihedral_outboard     = 0.
    segment.sweeps.quarter_chord  = 45. * Units.deg
    segment_airfoil               = SUAVE.Components.Wings.Airfoils.Airfoil()
    segment_airfoil.coordinate_file  = 'sc20414.dat'     
    segment.append_airfoil(segment_airfoil)
    wing.Segments.append(segment)       
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_2'
    segment.percent_span_location = 4.6/5.2
    segment.twist                 = 10. * Units.deg
    segment.root_chord_percent    = 2.3/6.2
    segment.dihedral_outboard     = 10. * Units.deg
    segment.sweeps.quarter_chord  = 65. * Units.deg
    segment_airfoil               = SUAVE.Components.Wings.Airfoils.Airfoil()
    segment_airfoil.coordinate_file  = 'sc20414.dat'     
    segment.append_airfoil(segment_airfoil)
    wing.Segments.append(segment)                 

    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'canard'
    
    wing.aspect_ratio            = 10.18
    wing.sweeps.quarter_chord    = -25 * Units.deg
    wing.thickness_to_chord      = 0.1
    wing.taper                   = 0.1
    wing.span_efficiency         = 0.9
    wing.dihedral                = -10 * Units.deg
    
    wing.spans.projected         = 34.32   
    
    wing.chords.root             = 7.760 * Units.meter
    wing.chords.tip              = 0.782 * Units.meter
    wing.chords.mean_aerodynamic = 4.235 * Units.meter
    wing.total_length            = 10.
    
    wing.areas.reference         = 124.862 
    
    wing.twists.root             = 100.0 * Units.degrees
    wing.twists.tip              = -10.0 * Units.degrees
    
    wing.origin                  = [2.,0,0.2]
    wing.aerodynamic_center      = [0,0,0]
    
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True
    
    wing.dynamic_pressure_ratio  = 1.0
    
    wing_airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    wing_airfoil.coordinate_file = 'NACA65-203.dat'                  

    vehicle.append_component(wing)    
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'    
    
    wing.aspect_ratio            = 1.91      #
    wing.sweeps.quarter_chord    = 25 * Units.deg
    wing.thickness_to_chord      = 0.08
    wing.taper                   = 0.25
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 7.777      #    

    wing.chords.root             = 8.19
    wing.chords.tip              = 0.95
    wing.chords.mean_aerodynamic = 4.0
    wing.total_length            = 10.
    
    wing.areas.reference         = 27.316    #
    
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  
    
    wing.origin                  = [28.79,0,1.]
    wing.aerodynamic_center      = [0,0,0]    #[2,0,0]    
    
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
    fuselage.lengths.cabin         = 28.85 #44.0
    fuselage.lengths.total         = 38.02 #58.4
    fuselage.lengths.fore_space    = 6.
    fuselage.lengths.aft_space     = 5.    
    
    fuselage.width                 = 3.74 
    
    fuselage.heights.maximum       = 3.74  
    fuselage.heights.at_quarter_length          = 3.74 
    fuselage.heights.at_three_quarters_length   = 3.65 
    fuselage.heights.at_wing_root_quarter_chord = 3.74 

    fuselage.areas.side_projected  = 142.1948
    fuselage.areas.wetted          = 446.718
    fuselage.areas.front_projected = 12.57
    
    fuselage.effective_diameter    = 3.74 
    
    fuselage.differential_pressure = 5.0e4 * Units.pascal 
    
    fuselage.OpenVSP_values = Data() # VSP uses degrees directly
    
    fuselage.OpenVSP_values.nose = Data()
    fuselage.OpenVSP_values.nose.top = Data()
    fuselage.OpenVSP_values.nose.side = Data()
    fuselage.OpenVSP_values.nose.top.angle = 20.0
    fuselage.OpenVSP_values.nose.top.strength = 0.75
    fuselage.OpenVSP_values.nose.side.angle = 20.0
    fuselage.OpenVSP_values.nose.side.strength = 0.75  
    fuselage.OpenVSP_values.nose.TB_Sym = True
    fuselage.OpenVSP_values.nose.z_pos = -.01
    
    fuselage.OpenVSP_values.tail = Data()
    fuselage.OpenVSP_values.tail.top = Data()
    fuselage.OpenVSP_values.tail.side = Data()    
    fuselage.OpenVSP_values.tail.bottom = Data()
    fuselage.OpenVSP_values.tail.top.angle = 0.0
    fuselage.OpenVSP_values.tail.top.strength = 0.0     
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
        
    # ------------------------------------------------------------------
    #   Turbojet Network
    # ------------------------------------------------------------------    

    # instantiate the gas turbine network
    turbojet = SUAVE.Components.Energy.Networks.Turbojet_Super()
    turbojet.tag = 'turbofan' # tagged for use with OpenVSP

    # setup
    turbojet.number_of_engines = 4.0
    turbojet.engine_length     = 12.0
    turbojet.nacelle_diameter  = 1.3
    turbojet.inlet_diameter    = 1.1
    turbojet.areas             = Data()
    turbojet.areas.wetted      = 12.5*4.7*2. # 4.7 is outer perimeter on one side
    turbojet.origin            = [[25.,8.,-2.3],[25.,5.3,-2.3],[25.,-5.3,-2.3],[25.,-8.,-2.3]]

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
    #   Configuration with Simple Engines
    # ------------------------------------------------------------------
    configs = SUAVE.Components.Configs.Config.Container()

    config = SUAVE.Components.Configs.Config(vehicle)
    config.tag = 'simple_engine'
    config.propulsors.turbofan.OpenVSP_simple           = True
    config.fuselages.fuselage.OpenVSP_values.tail.z_pos = 0.01
    configs.append(config)
    
    write(config,config.tag)

    # ------------------------------------------------------------------
    #   Configuration with flow-through engines
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(vehicle)
    config.tag = 'full_engine'
    configs.append(config)
    
    write(config,config.tag)

    return configs