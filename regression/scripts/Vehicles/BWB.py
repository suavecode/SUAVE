# BWB.py
# 
# Created:  Aug 2014, E. Botero
# Modified: Jan 2019, W. Maier

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
    #   Main Wing
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    
    wing.aspect_ratio            = 5.86 #12.
    wing.sweeps.quarter_chord    = 15. * Units.deg
    wing.thickness_to_chord      = 0.14
    wing.taper                   = 0.1
    wing.span_efficiency         = 0.9
    wing.dihedral                = 3.0 * Units.degrees

    wing.spans.projected         = 39. #38.7298

    wing.chords.root             = 17#16.
    wing.chords.tip              = 1.
    wing.chords.mean_aerodynamic = (2./3.)*(wing.chords.root + wing.chords.root -(wing.chords.root*wing.chords.root)/(wing.chords.root+wing.chords.root))

    wing.areas.reference         = 259.4#125.0

    wing.twists.root             = 1.0 * Units.degrees
    wing.twists.tip              = -4.0 * Units.degrees

    wing.origin                  = [3.,0.,-.25]
    wing.aerodynamic_center      = [3,0,-.25]

    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True

    wing.dynamic_pressure_ratio  = 1.0
    
    # New stuff
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'fuselage_edge'
    segment.percent_span_location = 7./wing.spans.projected
    segment.twist                 = -2. * Units.deg
    segment.root_chord_percent    = .88 #0.8  #.8
    segment.dihedral_outboard     = 10. * Units.deg
    segment.sweeps.quarter_chord  = 40*Units.deg#70. * Units.deg
    
    #airfoil = SUAVE.Components.Wings.Airfoils.Airfoil()
    #airfoil.coordinate_file = 'NACA2412'#'/Users/emiliobotero/Dropbox/SUAVE/Workspace/NACA2412.dat' # Or enter a NACA number
    
    #wing.append_airfoil(airfoil)
    
    #segment.append_airfoil(airfoil)
    wing.Segments.append(segment)
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'Outboard'
    segment.percent_span_location = .3
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.35
    segment.dihedral_outboard     = 4. * Units.deg
    segment.sweeps.quarter_chord  = 20. * Units.deg
    
    #section = SUAVE.Components.Wings.Airfoils.Airfoil()
    #section.airfoil = 'my_airfoil.dat' # Or enter a NACA number
    
    #segment.append(section)
    wing.Segments.append(segment)    

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------

    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage_bwb'

    fuselage.fineness.nose         = 0.65
    fuselage.fineness.tail         = 0.5

    fuselage.lengths.nose          = 4.0
    fuselage.lengths.tail          = 4.0
    fuselage.lengths.cabin         = 12.0
    fuselage.lengths.total         = 22.
    fuselage.lengths.fore_space    = 1.
    fuselage.lengths.aft_space     = 1.    

    fuselage.width                 = 8.

    fuselage.heights.maximum                    = 3.8
    fuselage.heights.at_quarter_length          = 3.7
    fuselage.heights.at_three_quarters_length   = 2.5
    fuselage.heights.at_wing_root_quarter_chord = 4.0

    fuselage.areas.side_projected  = 100.
    fuselage.areas.wetted          = 400.
    fuselage.areas.front_projected = 40.

    R = (fuselage.heights.maximum-fuselage.width)/(fuselage.heights.maximum-fuselage.width)
    fuselage.effective_diameter    = (fuselage.width/2 + fuselage.heights.maximum/2.)*(64.-3.*R**4.)/(64.-16.*R**2.)

    fuselage.differential_pressure = 5.0e4 * Units.pascal # Maximum differential pressure

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
    thrust.total_design = 2*24000. * Units.N #Newtons

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

    print("sls thrust : ",turbofan.sealevel_static_thrust)
    print("engine length : ",turbofan.engine_length)

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