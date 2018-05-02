# Bell_X1.py
#
# Created:  May 2018, W. Maier
# Modified: 

""" setup file for Bell X-1 vehicle
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import numpy as np
import SUAVE
from SUAVE.Core import Units
from SUAVE.Methods.Propulsion.liquid_rocket_sizing import liquid_rocket_sizing

# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------
def vehicle_setup():
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Bell_X1'    

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
    config.tag = 'general'
    configs.append(config)

    return configs
