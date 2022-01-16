# Additional_Fuel_Aircraft.py
#
# Created:  Jan 2022, S. Claridge (taken from data originally in the Boeing 737 tutorial script)

""" setup file for the vehicle based on the Boeing 737 with a turboelectric hts ducted fan network
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
from SUAVE.Core import Units

from SUAVE.Components.Energy.Networks.Turboelectric_HTS_Ducted_Fan import Turboelectric_HTS_Ducted_Fan   

from SUAVE.Methods.Propulsion.ducted_fan_sizing import ducted_fan_sizing

from SUAVE.Methods.Propulsion.serial_HTS_turboelectric_sizing import serial_HTS_turboelectric_sizing

from copy import deepcopy

from SUAVE.Attributes.Solids.Copper import Copper

from SUAVE.Attributes.Gases import Air


# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():
    """This is the full physical definition of the vehicle, and is designed to be independent of the
    analyses that are selected."""
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Boeing_737-800'    
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # Vehicle level mass properties
    # The maximum takeoff gross weight is used by a number of methods, most notably the weight
    # method. However, it does not directly inform mission analysis.
    vehicle.mass_properties.max_takeoff               = 79015.8 * Units.kilogram 
    # The takeoff weight is used to determine the weight of the vehicle at the start of the mission
    vehicle.mass_properties.takeoff                   = 79015.8 * Units.kilogram   
    # Operating empty may be used by various weight methods or other methods. Importantly, it does
    # not constrain the mission analysis directly, meaning that the vehicle weight in a mission
    # can drop below this value if more fuel is needed than is available.
    vehicle.mass_properties.operating_empty           = 62746.4 * Units.kilogram 
    # The maximum zero fuel weight is also used by methods such as weights
    vehicle.mass_properties.max_zero_fuel    = 62732.0 * Units.kilogram
    # Cargo weight typically feeds directly into weights output and does not affect the mission
    vehicle.mass_properties.cargo                     = 10000.  * Units.kilogram   
    
    # Envelope properties
    # These values are typical FAR values for a transport of this type
    vehicle.envelope.ultimate_load = 3.75
    vehicle.envelope.limit_load    = 2.5

    # Vehicle level parameters
    # The vehicle reference area typically matches the main wing reference area 
    vehicle.reference_area         = 124.862 * Units['meters**2']  

    # Number of passengers, control settings, and accessories settings are used by the weights
    # methods
    vehicle.passengers             = 170
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "medium range"

    # ------------------------------------------------------------------        
    #  Landing Gear
    # ------------------------------------------------------------------ 
    
    # The settings here can be used for noise analysis, but are not used in this tutorial
    landing_gear = SUAVE.Components.Landing_Gear.Landing_Gear()
    landing_gear.tag = "main_landing_gear"
    
    landing_gear.main_tire_diameter = 1.12000 * Units.m
    landing_gear.nose_tire_diameter = 0.6858 * Units.m
    landing_gear.main_strut_length  = 1.8 * Units.m
    landing_gear.nose_strut_length  = 1.3 * Units.m
    landing_gear.main_units  = 2    # Number of main landing gear
    landing_gear.nose_units  = 1    # Number of nose landing gear
    landing_gear.main_wheels = 2    # Number of wheels on the main landing gear
    landing_gear.nose_wheels = 2    # Number of wheels on the nose landing gear      
    vehicle.landing_gear = landing_gear

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    # This main wing is approximated as a simple trapezoid. A segmented wing can also be created if
    # desired. Segmented wings appear in later tutorials, and a version of the 737 with segmented
    # wings can be found in the SUAVE testing scripts.
    
    # SUAVE allows conflicting geometric values to be set in terms of items such as aspect ratio
    # when compared with span and reference area. Sizing scripts may be used to enforce 
    # consistency if desired.
    
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    wing.aspect_ratio            = 10.18
    # Quarter chord sweep is used as the driving sweep in most of the low fidelity analysis methods.
    # If a different known value (such as leading edge sweep) is given, it should be converted to
    # quarter chord sweep and added here. In some cases leading edge sweep will be used directly as
    # well, and can be entered here too.

    wing.sweeps.quarter_chord    = 25 * Units.deg
    wing.thickness_to_chord      = 0.1
    wing.taper                   = 0.1
    wing.spans.projected         = 34.32 * Units.meter
    wing.chords.root             = 7.760 * Units.meter
    wing.chords.tip              = 0.782 * Units.meter
    wing.chords.mean_aerodynamic = 4.235 * Units.meter
    wing.areas.reference         = 124.862 * Units['meters**2']  
    wing.twists.root             = 4.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    wing.origin                  = [[13.61, 0, -1.27]] * Units.meter
    wing.vertical                = False
    wing.symmetric               = True

    # The high lift flag controls aspects of maximum lift coefficient calculations
    wing.high_lift               = True
    # The dynamic pressure ratio is used in stability calculations
    wing.dynamic_pressure_ratio  = 1.0
    
    # ------------------------------------------------------------------
    #   Main Wing Control Surfaces
    # ------------------------------------------------------------------
    
    # Information in this section is used for high lift calculations and when conversion to AVL
    # is desired.
    
    # Deflections will typically be specified separately in individual vehicle configurations.
    
    flap                       = SUAVE.Components.Wings.Control_Surfaces.Flap() 
    flap.tag                   = 'flap' 
    flap.span_fraction_start   = 0.20 
    flap.span_fraction_end     = 0.70   
    flap.deflection            = 0.0 * Units.degrees
    # Flap configuration types are used in computing maximum CL and noise
    flap.configuration_type    = 'double_slotted'
    flap.chord_fraction        = 0.30   
    wing.append_control_surface(flap)   
        
    slat                       = SUAVE.Components.Wings.Control_Surfaces.Slat() 
    slat.tag                   = 'slat' 
    slat.span_fraction_start   = 0.324 
    slat.span_fraction_end     = 0.963     
    slat.deflection            = 0.0 * Units.degrees
    slat.chord_fraction        = 0.1  	 
    wing.append_control_surface(slat)  
        
    aileron                       = SUAVE.Components.Wings.Control_Surfaces.Aileron() 
    aileron.tag                   = 'aileron' 
    aileron.span_fraction_start   = 0.7 
    aileron.span_fraction_end     = 0.963 
    aileron.deflection            = 0.0 * Units.degrees
    aileron.chord_fraction        = 0.16    
    wing.append_control_surface(aileron)    
    
    # Add to vehicle
    vehicle.append_component(wing)    

    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Horizontal_Tail()
    wing.tag = 'horizontal_stabilizer'
    
    wing.aspect_ratio            = 6.16     
    wing.sweeps.quarter_chord    = 40.0 * Units.deg
    wing.thickness_to_chord      = 0.08
    wing.taper                   = 0.2
    wing.spans.projected         = 14.2 * Units.meter
    wing.chords.root             = 4.7  * Units.meter
    wing.chords.tip              = 0.955 * Units.meter
    wing.chords.mean_aerodynamic = 3.0  * Units.meter
    wing.areas.reference         = 32.488   * Units['meters**2']  
    wing.twists.root             = 3.0 * Units.degrees
    wing.twists.tip              = 3.0 * Units.degrees  
    wing.origin                  = [[32.83 * Units.meter, 0 , 1.14 * Units.meter]]
    wing.vertical                = False 
    wing.symmetric               = True
    wing.dynamic_pressure_ratio  = 0.9  
    
    # Add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Vertical_Tail()
    wing.tag = 'vertical_stabilizer'    

    wing.aspect_ratio            = 1.91
    wing.sweeps.quarter_chord    = 25. * Units.deg
    wing.thickness_to_chord      = 0.08
    wing.taper                   = 0.25
    wing.spans.projected         = 7.777 * Units.meter
    wing.chords.root             = 8.19  * Units.meter
    wing.chords.tip              = 0.95  * Units.meter
    wing.chords.mean_aerodynamic = 4.0   * Units.meter
    wing.areas.reference         = 27.316 * Units['meters**2']  
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  
    wing.origin                  = [[28.79 * Units.meter, 0, 1.54 * Units.meter]] # meters
    wing.vertical                = True 
    wing.symmetric               = False
    # The t tail flag is used in weights calculations
    wing.t_tail                  = False
    wing.dynamic_pressure_ratio  = 1.0
        
    # Add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    
    # Number of coach seats is used in some weights methods
    fuselage.number_coach_seats    = vehicle.passengers
    # The seats abreast can be used along with seat pitch and the number of coach seats to
    # determine the length of the cabin if desired.
    fuselage.seats_abreast         = 6
    fuselage.seat_pitch            = 1     * Units.meter
    # Fineness ratios are used to determine VLM fuselage shape and sections to use in OpenVSP
    # output
    fuselage.fineness.nose         = 1.6
    fuselage.fineness.tail         = 2.
    # Nose and tail lengths are used in the VLM setup
    fuselage.lengths.nose          = 6.4   * Units.meter
    fuselage.lengths.tail          = 8.0   * Units.meter
    fuselage.lengths.total         = 38.02 * Units.meter
    # Fore and aft space are added to the cabin length if the fuselage is sized based on
    # number of seats
    fuselage.lengths.fore_space    = 6.    * Units.meter
    fuselage.lengths.aft_space     = 5.    * Units.meter
    fuselage.width                 = 3.74  * Units.meter
    fuselage.heights.maximum       = 3.74  * Units.meter
    fuselage.effective_diameter    = 3.74     * Units.meter
    fuselage.areas.side_projected  = 142.1948 * Units['meters**2'] 
    fuselage.areas.wetted          = 446.718  * Units['meters**2'] 
    fuselage.areas.front_projected = 12.57    * Units['meters**2'] 
    # Maximum differential pressure between the cabin and the atmosphere
    fuselage.differential_pressure = 5.0e4 * Units.pascal
    
    # Heights at different longitudinal locations are used in stability calculations and
    # in output to OpenVSP
    fuselage.heights.at_quarter_length          = 3.74 * Units.meter
    fuselage.heights.at_three_quarters_length   = 3.65 * Units.meter
    fuselage.heights.at_wing_root_quarter_chord = 3.74 * Units.meter
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
    # ------------------------------------------------------------------
    #   Nacelles
    # ------------------------------------------------------------------ 
    nacelle                       = SUAVE.Components.Nacelles.Nacelle()
    nacelle.tag                   = 'nacelle_1'
    nacelle.length                = 2.71
    nacelle.inlet_diameter        = 1.90
    nacelle.diameter              = 2.05
    nacelle.areas.wetted          = 1.1*np.pi*nacelle.diameter*nacelle.length
    nacelle.origin                = [[13.72, -4.86,-1.9]]
    nacelle.flow_through          = True  
    nacelle_airfoil               = SUAVE.Components.Airfoils.Airfoil() 
    nacelle_airfoil.naca_4_series_airfoil = '2410'
    nacelle.append_airfoil(nacelle_airfoil)

    nacelle_2                     = deepcopy(nacelle)
    nacelle_2.tag                 = 'nacelle_2'
    nacelle_2.origin              = [[13.72, 4.86,-1.9]]
    
    vehicle.append_component(nacelle)  
    vehicle.append_component(nacelle_2)     

    # ------------------------------------------------------------------
    #   Turboelectric HTS Ducted Fan Network 
    # ------------------------------------------------------------------    
    
    # Instantiate the Turboelectric HTS Ducted Fan Network 
    # This also instantiates the component parts of the efan network, then below each part has its properties modified so they are no longer the default properties as created here at instantiation. 
    efan = Turboelectric_HTS_Ducted_Fan()
    efan.tag = 'turbo_fan'

    # Outline of Turboelectric drivetrain components. These are populated below.
    # 1. Propulsor     Ducted_fan
    #   1.1 Ram
    #   1.2 Inlet Nozzle
    #   1.3 Fan Nozzle
    #   1.4 Fan
    #   1.5 Thrust
    # 2. Motor         
    # 3. Powersupply   
    # 4. ESC           
    # 5. Rotor         
    # 6. Lead          
    # 7. CCS           
    # 8. Cryocooler    
    # 9. Heat Exchanger
    # The components are then sized

    # ------------------------------------------------------------------
    #Component 1 - Ducted Fan
    
    efan.ducted_fan                    = SUAVE.Components.Energy.Networks.Ducted_Fan()
    efan.ducted_fan.tag                = 'ducted_fan'
    efan.ducted_fan.number_of_engines  = 12.
    efan.number_of_engines             = efan.ducted_fan.number_of_engines
    efan.ducted_fan.engine_length      = 1.1            * Units.meter

    # Positioning variables for the propulsor locations - 
    xStart = 15.0
    xSpace = 1.0
    yStart = 3.0
    ySpace = 1.8
    efan.ducted_fan.origin =   [    [xStart+xSpace*5, -(yStart+ySpace*5), -2.0],
                                    [xStart+xSpace*4, -(yStart+ySpace*4), -2.0],
                                    [xStart+xSpace*3, -(yStart+ySpace*3), -2.0],
                                    [xStart+xSpace*2, -(yStart+ySpace*2), -2.0],
                                    [xStart+xSpace*1, -(yStart+ySpace*1), -2.0],
                                    [xStart+xSpace*0, -(yStart+ySpace*0), -2.0],
                                    [xStart+xSpace*5,  (yStart+ySpace*5), -2.0],
                                    [xStart+xSpace*4,  (yStart+ySpace*4), -2.0],
                                    [xStart+xSpace*3,  (yStart+ySpace*3), -2.0],
                                    [xStart+xSpace*2,  (yStart+ySpace*2), -2.0],
                                    [xStart+xSpace*1,  (yStart+ySpace*1), -2.0],
                                    [xStart+xSpace*0,  (yStart+ySpace*0), -2.0]     ] # meters 

    # copy the ducted fan details to the turboelectric ducted fan network to enable drag calculations
    efan.engine_length      = efan.ducted_fan.engine_length   
    efan.origin             = efan.ducted_fan.origin


    # working fluid
    efan.ducted_fan.working_fluid = SUAVE.Attributes.Gases.Air()
    
    # ------------------------------------------------------------------
    #   Component 1.1 - Ram
    
    # to convert freestream static to stagnation quantities
    # instantiate
    ram = SUAVE.Components.Energy.Converters.Ram()
    ram.tag = 'ram'
    
    # add to the network
    efan.ducted_fan.append(ram)

    # ------------------------------------------------------------------
    #  Component 1.2 - Inlet Nozzle
    
    # instantiate
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag = 'inlet_nozzle'
    
    # setup
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio        = 0.98
    
    # add to network
    efan.ducted_fan.append(inlet_nozzle)

    # ------------------------------------------------------------------
    #  Component 1.3 - Fan Nozzle
    
    # instantiate
    fan_nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    fan_nozzle.tag = 'fan_nozzle'

    # setup
    fan_nozzle.polytropic_efficiency = 0.95
    fan_nozzle.pressure_ratio        = 0.99    
    
    # add to network
    efan.ducted_fan.append(fan_nozzle)
    
    # ------------------------------------------------------------------
    #  Component 1.4 - Fan
    
    # instantiate
    fan = SUAVE.Components.Energy.Converters.Fan()
    fan.tag = 'fan'

    # setup
    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio        = 1.7    
    
    # add to network
    efan.ducted_fan.append(fan)

    # ------------------------------------------------------------------
    # Component 1.5 : thrust

    # To compute the thrust
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='compute_thrust'

    # total design thrust (includes all the propulsors)
    thrust.total_design             = 2.*24000. * Units.N #Newtons

    # design sizing conditions
    altitude      = 35000.0*Units.ft
    mach_number   =     0.78 
    isa_deviation =     0.
    
    # add to network
    efan.ducted_fan.thrust = thrust
    # ------------------------------------------------------------------
    # Component 2 : HTS motor
    
    efan.motor = SUAVE.Components.Energy.Converters.Motor_Lo_Fid()
    efan.motor.tag = 'motor'
    # number_of_motors is not used as the motor count is assumed to match the engine count

    # Set the origin of each motor to match its ducted fan
    efan.motor.origin = efan.ducted_fan.origin
    efan.motor.gear_ratio         = 1.0
    efan.motor.gearbox_efficiency = 1.0
    efan.motor.motor_efficiency   = 0.96

    # ------------------------------------------------------------------
    #  Component 3 - Powersupply
    
    efan.powersupply                        = SUAVE.Components.Energy.Converters.Turboelectric()
    efan.powersupply.tag                    = 'powersupply'

    efan.number_of_powersupplies            = 2.
    efan.powersupply.propellant             = SUAVE.Attributes.Propellants.Jet_A()
    efan.powersupply.oxidizer               = Air()
    efan.powersupply.number_of_engines      = 2.0                   # number of turboelectric machines, not propulsors
    efan.powersupply.efficiency             = .37                   # Approximate average gross efficiency across the product range.
    efan.powersupply.volume                 = 2.36    *Units.m**3.  # 3m long from RB211 datasheet. 1m estimated radius.
    efan.powersupply.rated_power            = 37400.0 *Units.kW
    efan.powersupply.mass_properties.mass   = 2500.0  *Units.kg     # 2.5 tonnes from Rolls Royce RB211 datasheet 2013.
    efan.powersupply.specific_power         = efan.powersupply.rated_power/efan.powersupply.mass_properties.mass
    efan.powersupply.mass_density           = efan.powersupply.mass_properties.mass /efan.powersupply.volume 

    # ------------------------------------------------------------------
    #  Component 4 - Electronic Speed Controller (ESC)
    
    efan.esc = SUAVE.Components.Energy.Distributors.HTS_DC_Supply()     # Could make this where the ESC is defined as a Siemens SD104
    efan.esc.tag = 'esc'

    efan.esc.efficiency             =   0.95                 # Siemens SD104 SiC Power Electronicss reported to be this efficient

    # ------------------------------------------------------------------
    #  Component 5 - HTS rotor (part of the propulsor motor)
    
    efan.rotor = SUAVE.Components.Energy.Converters.Motor_HTS_Rotor()
    efan.rotor.tag = 'rotor'

    efan.rotor.temperature              =    50.0       # [K]
    efan.rotor.skin_temp                =   300.0       # [K]       Temp of rotor outer surface is not ambient
    efan.rotor.current                  =  1000.0       # [A]       Most of the cryoload will scale with this number if not using HTS Dynamo
    efan.rotor.resistance               =     0.0001    # [ohm]     20 x 100 nOhm joints should be possible (2uOhm total) so 1mOhm is an overestimation.
    efan.rotor.number_of_engines        = efan.ducted_fan.number_of_engines      
    efan.rotor.length                   =     0.573     * Units.meter       # From paper: DOI:10.2514/6.2019-4517 Would be good to estimate this from power instead.
    efan.rotor.diameter                 =     0.310     * Units.meter       # From paper: DOI:10.2514/6.2019-4517 Would be good to estimate this from power instead.
    rotor_end_area                      = np.pi*(efan.rotor.diameter/2.0)**2.0
    rotor_end_circumference             = np.pi*efan.rotor.diameter
    efan.rotor.surface_area             = 2.0 * rotor_end_area + efan.rotor.length*rotor_end_circumference
    efan.rotor.R_value                  =   125.0                           # [K.m2/W]  2.0 W/m2 based on experience at Robinson Research

    # ------------------------------------------------------------------
    #  Component 6 - Copper Supply Leads of propulsion motor rotors
    
    efan.lead = SUAVE.Components.Energy.Distributors.Cryogenic_Lead()
    efan.lead.tag = 'lead'

    copper = Copper()
    efan.lead.cold_temp                 = efan.rotor.temperature   # [K]
    efan.lead.hot_temp                  = efan.rotor.skin_temp     # [K]
    efan.lead.current                   = efan.rotor.current       # [A]
    efan.lead.length                    = 0.3                      # [m]
    efan.lead.material                  = copper
    efan.leads                          = efan.ducted_fan.number_of_engines * 2.0      # Each motor has two leads to make a complete circuit

    # ------------------------------------------------------------------
    #  Component 7 - Rotor Constant Current Supply (CCS)
    
    efan.ccs = SUAVE.Components.Energy.Distributors.HTS_DC_Supply()
    efan.ccs.tag = 'ccs'

    efan.ccs.efficiency             =   0.95               # Siemens SD104 SiC Power Electronics reported to be this efficient
    # ------------------------------------------------------------------
    #  Component 8 - Cryocooler, to cool the HTS Rotor
 
    efan.cryocooler = SUAVE.Components.Energy.Cooling.Cryocooler()
    efan.cryocooler.tag = 'cryocooler'

    efan.cryocooler.cooler_type        = 'GM'
    efan.cryocooler.min_cryo_temp      = efan.rotor.temperature     # [K]
    efan.cryocooler.ambient_temp       = 300.0                      # [K]

    # ------------------------------------------------------------------
    #  Component 9 - Cryogenic Heat Exchanger, to cool the HTS Rotor
    efan.heat_exchanger = SUAVE.Components.Energy.Cooling.Cryogenic_Heat_Exchanger()
    efan.heat_exchanger.tag = 'heat_exchanger'

    efan.heat_exchanger.cryogen                         = SUAVE.Attributes.Cryogens.Liquid_H2()
    efan.heat_exchanger.cryogen_inlet_temperature       =     20.0                  # [K]
    efan.heat_exchanger.cryogen_outlet_temperature      = efan.rotor.temperature    # [K]
    efan.heat_exchanger.cryogen_pressure                = 100000.0                  # [Pa]
    efan.heat_exchanger.cryogen_is_fuel                 =      0.0

    # Sizing Conditions. The cryocooler may have greater power requirement at low altitude as the cooling requirement may be static during the flight but the ambient temperature may change.
    cryo_temp       =  50.0     # [K]
    amb_temp        = 300.0     # [K]

    # ------------------------------------------------------------------
    # Powertrain Sizing

    ducted_fan_sizing(efan.ducted_fan,mach_number,altitude)

    serial_HTS_turboelectric_sizing(efan,mach_number,altitude, cryo_cold_temp = cryo_temp, cryo_amb_temp = amb_temp)

    # add turboelectric network to the vehicle 
    vehicle.append_component(efan)

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    return vehicle


def configs_setup(vehicle):
    """This function sets up vehicle configurations for use in different parts of the mission.
    Here, this is mostly in terms of high lift settings."""
    
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
    config.wings['main_wing'].control_surfaces.flap.deflection = 20. * Units.deg
    config.wings['main_wing'].control_surfaces.slat.deflection = 25. * Units.deg
    # A max lift coefficient factor of 1 is the default, but it is highlighted here as an option
    config.max_lift_coefficient_factor    = 1.

    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Cutback Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cutback'
    config.wings['main_wing'].control_surfaces.flap.deflection = 20. * Units.deg
    config.wings['main_wing'].control_surfaces.slat.deflection = 20. * Units.deg
    config.max_lift_coefficient_factor    = 1.

    configs.append(config)    

    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------

    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'landing'

    config.wings['main_wing'].control_surfaces.flap.deflection = 30. * Units.deg
    config.wings['main_wing'].control_surfaces.slat.deflection = 25. * Units.deg  
    config.max_lift_coefficient_factor    = 1. 

    configs.append(config)

    # ------------------------------------------------------------------
    #   Short Field Takeoff Configuration
    # ------------------------------------------------------------------ 

    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'short_field_takeoff'
    
    config.wings['main_wing'].control_surfaces.flap.deflection = 20. * Units.deg
    config.wings['main_wing'].control_surfaces.slat.deflection = 20. * Units.deg
    config.max_lift_coefficient_factor    = 1. 
  
    configs.append(config)

    return configs
