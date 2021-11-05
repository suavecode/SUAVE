# Embraer_190
#
# Created:  Feb 2017, M. Vegh (data taken from Embraer_E190_constThr/mission_Embraer_E190_constThr, and Regional_Jet_Optimization/Vehicles2.py), takeoff_field_length/takeoff_field_length.py, landing_field_length/landing_field_length.py
# Modified: Mar 2020, M. Clarke
#           May 2020, E. Botero
#           Oct 2021, M. Clarke


""" setup file for the E190 vehicle
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
from SUAVE.Core import Units
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform, segment_properties

from copy import deepcopy

# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Embraer_E190AR'

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------

    # mass properties (http://www.embraercommercialaviation.com/AircraftPDF/E190_Weights.pdf)
    vehicle.mass_properties.max_takeoff               = 51800.   # kg
    vehicle.mass_properties.operating_empty           = 27837.   # kg
    vehicle.mass_properties.takeoff                   = 51800.   # kg
    vehicle.mass_properties.max_zero_fuel             = 40900.   # kg
    vehicle.mass_properties.max_payload               = 13063.   # kg
    vehicle.mass_properties.max_fuel                  = 12971.   # kg
    vehicle.mass_properties.cargo                     =     0.0  # kg

    vehicle.mass_properties.center_of_gravity         = [[16.8, 0, 1.6]]
    vehicle.mass_properties.moments_of_inertia.tensor = [[10 ** 5, 0, 0],[0, 10 ** 6, 0,],[0,0, 10 ** 7]] 

    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 92.
    vehicle.passengers             = 106
    vehicle.systems.control        = "fully powered"
    vehicle.systems.accessories    = "medium range"


    # ------------------------------------------------------------------
    #   Main Wing
    # ------------------------------------------------------------------
    wing                         = SUAVE.Components.Wings.Main_Wing()
    wing.tag                     = 'main_wing'
    wing.areas.reference         = 92.0
    wing.aspect_ratio            = 8.4
    wing.chords.root             = 6.2
    wing.chords.tip              = 1.44
    wing.sweeps.quarter_chord    = 23.0 * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.28
    wing.dihedral                = 5.00 * Units.deg
    wing.spans.projected         = 28.72
    wing.origin                  = [[13.0,0,-1.50]]
    wing.vertical                = False
    wing.symmetric               = True       
    wing.high_lift               = True
    wing.areas.exposed           = 0.80 * wing.areas.wetted        
    wing.twists.root             = 2.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees    
    wing.dynamic_pressure_ratio  = 1.0
    
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'root'
    segment.percent_span_location = 0.0
    segment.twist                 = 4. * Units.deg
    segment.root_chord_percent    = 1.
    segment.thickness_to_chord    = .11
    segment.dihedral_outboard     = 5. * Units.degrees
    segment.sweeps.quarter_chord  = 20.6 * Units.degrees
    wing.Segments.append(segment)    
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'yehudi'
    segment.percent_span_location = 0.348
    segment.twist                 = (4. - segment.percent_span_location*4.) * Units.deg
    segment.root_chord_percent    = 0.60
    segment.thickness_to_chord    = .11
    segment.dihedral_outboard     = 4 * Units.degrees
    segment.sweeps.quarter_chord  = 24.1 * Units.degrees
    wing.Segments.append(segment)
    
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'section_2'
    segment.percent_span_location = 0.961
    segment.twist                 = (4. - segment.percent_span_location*4.) * Units.deg
    segment.root_chord_percent    = 0.25
    segment.thickness_to_chord    = .11
    segment.dihedral_outboard     = 70. * Units.degrees
    segment.sweeps.quarter_chord  = 50. * Units.degrees
    wing.Segments.append(segment)

    segment = SUAVE.Components.Wings.Segment() 
    segment.tag                   = 'Tip'
    segment.percent_span_location = 1.
    segment.twist                 = (4. - segment.percent_span_location*4.) * Units.deg
    segment.root_chord_percent    = 0.070
    segment.thickness_to_chord    = .11
    segment.dihedral_outboard     = 0.
    segment.sweeps.quarter_chord  = 0.
    wing.Segments.append(segment)       
    
    # Fill out more segment properties automatically
    wing = segment_properties(wing)        

    # control surfaces -------------------------------------------
    flap                       = SUAVE.Components.Wings.Control_Surfaces.Flap() 
    flap.tag                   = 'flap' 
    flap.span_fraction_start   = 0.11
    flap.span_fraction_end     = 0.85
    flap.deflection            = 0.0 * Units.deg 
    flap.chord_fraction        = 0.28    
    flap.configuration_type    = 'double_slotted'
    wing.append_control_surface(flap)   
        
    slat                       = SUAVE.Components.Wings.Control_Surfaces.Slat()
    slat.tag                   = 'slat' 
    slat.span_fraction_start   = 0.324 
    slat.span_fraction_end     = 0.963     
    slat.deflection            = 1.0 * Units.deg 
    slat.chord_fraction        = 0.1   
    wing.append_control_surface(slat) 
    
    wing                         = wing_planform(wing)
    
    wing.areas.exposed           = 0.80 * wing.areas.wetted
    wing.twists.root             = 2.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees    
    wing.dynamic_pressure_ratio  = 1.0   

    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Horizontal_Tail()
    wing.tag = 'horizontal_stabilizer'
    wing.areas.reference         = 26.0
    wing.aspect_ratio            = 5.5
    wing.sweeps.quarter_chord    = 34.5 * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.11
    wing.dihedral                = 8.4 * Units.degrees
    wing.origin                  = [[31,0,0.44]]
    wing.vertical                = False
    wing.symmetric               = True       
    wing.high_lift               = False  
    wing                         = wing_planform(wing)
    wing.areas.exposed           = 0.9 * wing.areas.wetted 
    wing.twists.root             = 2.0 * Units.degrees
    wing.twists.tip              = 2.0 * Units.degrees    
    wing.dynamic_pressure_ratio  = 0.90

    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Vertical_Tail()
    wing.tag = 'vertical_stabilizer'
    wing.areas.reference         = 16.0
    wing.aspect_ratio            =  1.7
    wing.sweeps.quarter_chord    = 35. * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.31
    wing.dihedral                = 0.00
    wing.origin                  = [[30.4,0,1.675]]
    wing.vertical                = True
    wing.symmetric               = False       
    wing.high_lift               = False
    wing                         = wing_planform(wing)
    wing.areas.exposed           = 0.9 * wing.areas.wetted
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees    
    wing.dynamic_pressure_ratio  = 1.00
    
    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------

    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag    = 'fuselage'
    fuselage.origin = [[0,0,0]]
    fuselage.number_coach_seats    = vehicle.passengers
    fuselage.seats_abreast         = 4
    fuselage.seat_pitch            = 30. * Units.inches

    fuselage.fineness.nose         = 1.28
    fuselage.fineness.tail         = 3.48

    fuselage.lengths.nose          = 6.0
    fuselage.lengths.tail          = 9.0
    fuselage.lengths.cabin         = 21.24
    fuselage.lengths.total         = 36.24
    fuselage.lengths.fore_space    = 0.
    fuselage.lengths.aft_space     = 0.

    fuselage.width                 = 3.01 * Units.meters

    fuselage.heights.maximum       = 3.35    
    fuselage.heights.at_quarter_length          = 3.35 
    fuselage.heights.at_three_quarters_length   = 3.35 
    fuselage.heights.at_wing_root_quarter_chord = 3.35 

    fuselage.areas.side_projected  = 239.20
    fuselage.areas.wetted          = 327.01
    fuselage.areas.front_projected = 8.0110

    fuselage.effective_diameter    = 3.18

    fuselage.differential_pressure = 10**5 * Units.pascal    # Maximum differential pressure

    # add to vehicle
    vehicle.append_component(fuselage)

    # -----------------------------------------------------------------
    # Design the Nacelle
    # ----------------------------------------------------------------- 
    nacelle                       = SUAVE.Components.Nacelles.Nacelle()
    nacelle.diameter              = 2.05
    nacelle.length                = 2.71
    nacelle.tag                   = 'nacelle_1'
    nacelle.inlet_diameter        = 2.0
    nacelle.origin                = [[12.0,4.38,-2.1]]
    Awet                          = 1.1*np.pi*nacelle.diameter*nacelle.length # 1.1 is simple coefficient
    nacelle.areas.wetted          = Awet 
    nacelle_airfoil               = SUAVE.Components.Airfoils.Airfoil() 
    nacelle_airfoil.naca_4_series_airfoil = '2410'
    nacelle.append_airfoil(nacelle_airfoil) 

    nacelle_2                     = deepcopy(nacelle)
    nacelle_2.tag                 = 'nacelle_2'
    nacelle_2.origin              = [[12.0,-4.38,-2.1]]
    
    vehicle.append_component(nacelle)   
    vehicle.append_component(nacelle_2)   
    
    
    # ------------------------------------------------------------------
    #  Turbofan Network
    # ------------------------------------------------------------------ 
    #initialize the gas turbine network
    gt_engine                   = SUAVE.Components.Energy.Networks.Turbofan()
    gt_engine.tag               = 'turbofan'
    gt_engine.origin            = [[12.0,4.38,-2.1],[12.0,-4.38,-2.1]] 
    gt_engine.engine_length     = 2.71    
    gt_engine.number_of_engines = 2.0
    gt_engine.bypass_ratio      = 5.4   
  
    #set the working fluid for the network
    working_fluid               = SUAVE.Attributes.Gases.Air()

    #add working fluid to the network
    gt_engine.working_fluid     = working_fluid


    #Component 1 : ram,  to convert freestream static to stagnation quantities
    ram           = SUAVE.Components.Energy.Converters.Ram()
    ram.tag       = 'ram'
    #add ram to the network
    gt_engine.ram = ram


    #Component 2 : inlet nozzle
    inlet_nozzle                       = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag                   = 'inlet nozzle'
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio        = 0.98
    #add inlet nozzle to the network
    gt_engine.inlet_nozzle             = inlet_nozzle


    #Component 3 :low pressure compressor    
    low_pressure_compressor                       = SUAVE.Components.Energy.Converters.Compressor()    
    low_pressure_compressor.tag                   = 'lpc'
    low_pressure_compressor.polytropic_efficiency = 0.91
    low_pressure_compressor.pressure_ratio        = 1.9    
    #add low pressure compressor to the network    
    gt_engine.low_pressure_compressor             = low_pressure_compressor

    #Component 4 :high pressure compressor  
    high_pressure_compressor                       = SUAVE.Components.Energy.Converters.Compressor()    
    high_pressure_compressor.tag                   = 'hpc'
    high_pressure_compressor.polytropic_efficiency = 0.91
    high_pressure_compressor.pressure_ratio        = 10.0   
    #add the high pressure compressor to the network    
    gt_engine.high_pressure_compressor             = high_pressure_compressor

    #Component 5 :low pressure turbine  
    low_pressure_turbine                        = SUAVE.Components.Energy.Converters.Turbine()   
    low_pressure_turbine.tag                    ='lpt'
    low_pressure_turbine.mechanical_efficiency  = 0.99
    low_pressure_turbine.polytropic_efficiency  = 0.93
    #add low pressure turbine to the network     
    gt_engine.low_pressure_turbine              = low_pressure_turbine

    #Component 5 :high pressure turbine  
    high_pressure_turbine                       = SUAVE.Components.Energy.Converters.Turbine()   
    high_pressure_turbine.tag                   ='hpt'
    high_pressure_turbine.mechanical_efficiency = 0.99
    high_pressure_turbine.polytropic_efficiency = 0.93
    #add the high pressure turbine to the network    
    gt_engine.high_pressure_turbine             = high_pressure_turbine 

    #Component 6 :combustor  
    combustor                           = SUAVE.Components.Energy.Converters.Combustor()   
    combustor.tag                       = 'Comb'
    combustor.efficiency                = 0.99 
    combustor.alphac                    = 1.0     
    combustor.turbine_inlet_temperature = 1500
    combustor.pressure_ratio            = 0.95
    combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()    
    #add the combustor to the network    
    gt_engine.combustor                 = combustor

    #Component 7 :core nozzle
    core_nozzle                       = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    core_nozzle.tag                   = 'core nozzle'
    core_nozzle.polytropic_efficiency = 0.95
    core_nozzle.pressure_ratio        = 0.99    
    #add the core nozzle to the network    
    gt_engine.core_nozzle             = core_nozzle

    #Component 8 :fan nozzle
    fan_nozzle                       = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    fan_nozzle.tag                   = 'fan nozzle'
    fan_nozzle.polytropic_efficiency = 0.95
    fan_nozzle.pressure_ratio        = 0.99
    #add the fan nozzle to the network
    gt_engine.fan_nozzle             = fan_nozzle

    #Component 9 : fan   
    fan                       = SUAVE.Components.Energy.Converters.Fan()   
    fan.tag                   = 'fan'
    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio        = 1.7  
    #add the fan to the network
    gt_engine.fan             = fan    

    #Component 10 : thrust (to compute the thrust)
    thrust     = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='compute_thrust'
    #total design thrust (includes all the engines)
    thrust.total_design   = 37278.0* Units.N #Newtons

    #design sizing conditions
    altitude         = 35000.0*Units.ft
    mach_number      = 0.78 
    isa_deviation    = 0.
    # add thrust to the network
    gt_engine.thrust = thrust

    #size the turbofan
    turbofan_sizing(gt_engine,mach_number,altitude)   

    # add  gas turbine network gt_engine to the vehicle
    vehicle.append_component(gt_engine)      
    
    fuel                                  = SUAVE.Components.Physical_Component()
    vehicle.fuel                          = fuel
    fuel.mass_properties.mass             = vehicle.mass_properties.max_takeoff-vehicle.mass_properties.max_fuel
    fuel.origin                           = vehicle.wings.main_wing.mass_properties.center_of_gravity     
    fuel.mass_properties.center_of_gravity= vehicle.wings.main_wing.aerodynamic_center
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
    config.wings['main_wing'].control_surfaces.flap.deflection  = 20. * Units.deg
    config.wings['main_wing'].control_surfaces.slat.deflection  = 25. * Units.deg
    config.V2_VS_ratio = 1.21
    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------

    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'landing'
    config.wings['main_wing'].control_surfaces.flap.deflection  = 30. * Units.deg
    config.wings['main_wing'].control_surfaces.slat.deflection  = 25. * Units.deg
    config.Vref_VS_ratio = 1.23
    configs.append(config)   
     
    # ------------------------------------------------------------------
    #   Short Field Takeoff Configuration
    # ------------------------------------------------------------------ 

    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'short_field_takeoff'    
    config.wings['main_wing'].control_surfaces.flap.deflection  = 20. * Units.deg
    config.wings['main_wing'].control_surfaces.slat.deflection  = 25. * Units.deg
    config.V2_VS_ratio = 1.21
    
    # payload?
    
    configs.append(config)

    # done!
    return configs
