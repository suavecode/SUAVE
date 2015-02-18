# test_mission_SUGAR_Ray.py
#
# Created: Tim Momose, November 2014


""" setup file for a mission with the SUGAR Ray BWB concept
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Attributes import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

# the analysis functions
from the_aircraft_function_copy import the_aircraft_function
from plot_mission_copy import plot_mission


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    # define the problem
    vehicle, mission = full_setup()
    
    # run the problem
    results = the_aircraft_function(vehicle,mission)
    
    # plot the new results
    plot_mission(vehicle,mission,results,'bo-')    
    
    # load older results
    #save_results(results)
    #old_results = load_results()
    
    # plt the old results
    #plot_mission(vehicle,mission,old_results,'k-')
    
    ## check the results
    #check_results(results,old_results)
    
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    vehicle = vehicle_setup()
    mission = mission_setup(vehicle)
    
    return vehicle, mission


# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'SUGAR Ray BWB'    
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff               = 182500. * Units.lb
    vehicle.mass_properties.operating_empty           = 110493. * Units.lb
    vehicle.mass_properties.takeoff                   = 182500. * Units.lb
    vehicle.mass_properties.max_zero_fuel             = 0.9 * vehicle.mass_properties.max_takeoff 
    #vehicle.mass_properties.cargo                     = 10000.  * Units.kilogram   
    
    vehicle.mass_properties.center_of_gravity         = [478.4 * Units.inches, 0, 0] 
    vehicle.mass_properties.moments_of_inertia.tensor = [[10 ** 5, 0, 0],[0, 10 ** 6, 0,],[0,0, 10 ** 7]] # Not Correct
    
    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area         = 2900. * Units.feet**2.
    vehicle.passengers             = 155.
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "medium range"
    
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Main Wing'
    
    wing.aspect_ratio            = 6.865
    wing.sweep                   = 30. * Units.deg
    wing.thickness_to_chord      = 0.13
    wing.taper                   = 73.6/727.8
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 1936.8 * Units.inches    
    
    wing.chords.root             = 727.8 * Units.inches
    wing.chords.tip              = 73.6 * Units.inches
    wing.chords.mean_aerodynamic = 489.7 * Units.inches
    
    wing.areas.reference         = 2900. * Units.feet**2.
    wing.areas.gross             = 4136. * Units.feet**2.
    wing.areas.wetted            = 2.0 * wing.areas.gross
    wing.areas.exposed           = wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.wetted
    
    wing.transition_x_lower      = 0.6
    wing.transition_x_upper      = 0.5
    
    wing.twists.root             = 1.0 * Units.degrees
    wing.twists.tip              = -4.0 * Units.degrees
    
    wing.dihedral                = 3.0 * Units.degrees
    
    wing.origin                  = [102.8*Units.inches,0,0]
    xcg = vehicle.mass_properties.center_of_gravity[0]
    xo  = wing.origin[0]
    mac = wing.chords.mean_aerodynamic
    acw =xcg-xo-mac*0.15
    wing.aerodynamic_center      = [acw,0,0] 
    
    wing.vertical                = False
    wing.symmetric               = True
    
    wing.dynamic_pressure_ratio  = 1.0
    
    # add to vehicle
    vehicle.append_component(wing)
    
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertical Stabilizer'    
    
    wing.aspect_ratio            = 1.705
    wing.sweep                   = 42. * Units.deg
    wing.thickness_to_chord      = 0.08
    wing.taper                   = 0.366
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 10.22 * Units.feet    

    wing.chords.root             = 129.23 * Units.inches
    wing.chords.tip              = 44.9 * Units.inches
    wing.chords.mean_aerodynamic = 101.3 * Units.inches
    
    wing.areas.reference         = 2. * 90.8 * Units.feet**2.
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.wetted
    
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  
    
    wing.origin                  = [745.*Units.inches,0,0]
    wing.aerodynamic_center      = [60.*Units.inches,0,0]    
    
    wing.vertical                = True 
    wing.symmetric               = False
    wing.t_tail                  = False
    
    wing.dynamic_pressure_ratio  = 1.0
        
    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #   Dummy Horizontal Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Horizontal Stabilizer'    
    
    wing.aspect_ratio            = 1.705
    wing.sweep                   = 42. * Units.deg
    wing.thickness_to_chord      = 0.08
    wing.taper                   = 0.366
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 10.22 * Units.feet    

    wing.chords.root             = 129.23 * Units.inches
    wing.chords.tip              = 44.9 * Units.inches
    wing.chords.mean_aerodynamic = 101.3 * Units.inches
    
    wing.areas.reference         = 0.01
    wing.areas.wetted            = 0.01#2.0 * wing.areas.reference 
    wing.areas.exposed           = 0.#0.8 * wing.areas.wetted
    wing.areas.affected          = 0.#0.6 * wing.areas.wetted
    
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  
    
    wing.origin                  = [745.*Units.inches,0,0]
    wing.aerodynamic_center      = [60.*Units.inches,0,0]    
    
    wing.vertical                = False
    wing.symmetric               = True
    wing.t_tail                  = False
    
    wing.dynamic_pressure_ratio  = 1.0
        
    # add to vehicle
    vehicle.append_component(wing)
    

    # ------------------------------------------------------------------
    #  Dummy Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'
    
    fuselage.number_coach_seats    = vehicle.passengers
    fuselage.seats_abreast         = 16
    fuselage.seat_pitch            = 1
    
    fuselage.fineness.nose         = 1.5  # not really
    fuselage.fineness.tail         = 2.   # not really
    
    fuselage.lengths.nose          = 3.
    fuselage.lengths.tail          = 4.
    fuselage.lengths.cabin         = 592. * Units.inches
    fuselage.lengths.total         = 888.5 * Units.inches 
    fuselage.lengths.fore_space    = 6.
    fuselage.lengths.aft_space     = 2.    
    
    fuselage.width                 = 500. * Units.inches
    
    fuselage.heights.maximum       = 4.    #
    fuselage.heights.at_quarter_length          = 4. # Not correct
    fuselage.heights.at_three_quarters_length   = 4. # Not correct
    fuselage.heights.at_vertical_root_quarter_chord = 4. # Not correct

    fuselage.areas.side_projected  = 80.
    fuselage.areas.wetted          = 0.
    fuselage.areas.front_projected = 1. #Necessary to get a parasite drag analysis
    
    fuselage.effective_diameter    = 4.0
    
    fuselage.differential_pressure = 10**5 * Units.pascal    # Maximum differential pressure
    
    # add to vehicle
    vehicle.append_component(fuselage)
    

    
    # ------------------------------------------------------------------
    #  Turbofan Network
    # ------------------------------------------------------------------    
    
    # USING THE B737 ENGINES FOR NOW BECAUSE WHAT I HAD WAS GIVING nan MACH NUMBERS AT THE INLET NOZZLE
    #initialize the gas turbine network
    gt_engine                   = SUAVE.Components.Energy.Networks.Turbofan_Network()
    gt_engine.tag               = 'Turbo Fan'
    
    gt_engine.number_of_engines = 3.0 #2.0
    gt_engine.thrust_design     = 27875. #18800 * Units.lbf #24000.0
    gt_engine.engine_length     = 2.5
    gt_engine.nacelle_diameter  = 1.580

    #set the working fluid for the network
    working_fluid               = SUAVE.Attributes.Gases.Air
    
    #add working fluid to the network
    gt_engine.working_fluid = working_fluid
    
    
    #Component 1 : ram,  to convert freestream static to stagnation quantities
    ram = SUAVE.Components.Energy.Converters.Ram()
    ram.tag = 'ram'
    
    #add ram to the network
    gt_engine.ram = ram
    
    
    #Component 2 : inlet nozzle
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag = 'inlet nozzle'

    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio        = 0.98 
    
    #add inlet nozzle to the network
    gt_engine.inlet_nozzle = inlet_nozzle
    
    
    #Component 3 :low pressure compressor    
    low_pressure_compressor = SUAVE.Components.Energy.Converters.Compressor()    
    low_pressure_compressor.tag = 'lpc'
    
    low_pressure_compressor.polytropic_efficiency = 0.91
    low_pressure_compressor.pressure_ratio        = 1.14    
    
    #add low pressure compressor to the network    
    gt_engine.low_pressure_compressor = low_pressure_compressor

    

    #Component 4 :high pressure compressor  
    high_pressure_compressor = SUAVE.Components.Energy.Converters.Compressor()    
    high_pressure_compressor.tag = 'hpc'
    
    high_pressure_compressor.polytropic_efficiency = 0.91
    high_pressure_compressor.pressure_ratio        = 13.415    
    
    #add the high pressure compressor to the network    
    gt_engine.high_pressure_compressor = high_pressure_compressor


    #Component 5 :low pressure turbine  
    low_pressure_turbine = SUAVE.Components.Energy.Converters.Turbine()   
    low_pressure_turbine.tag='lpt'
    
    low_pressure_turbine.mechanical_efficiency = 0.99
    low_pressure_turbine.polytropic_efficiency = 0.93     
    
    #add low pressure turbine to the network    
    gt_engine.low_pressure_turbine = low_pressure_turbine
      
    
    
    #Component 5 :high pressure turbine  
    high_pressure_turbine = SUAVE.Components.Energy.Converters.Turbine()   
    high_pressure_turbine.tag='hpt'

    high_pressure_turbine.mechanical_efficiency = 0.99
    high_pressure_turbine.polytropic_efficiency = 0.93     
    
    #add the high pressure turbine to the network    
    gt_engine.high_pressure_turbine = high_pressure_turbine 
      
    
    
    #Component 6 :combustor  
    combustor = SUAVE.Components.Energy.Converters.Combustor()   
    combustor.tag = 'Comb'
    
    combustor.efficiency                = 0.99 
    combustor.alphac                    = 1.0     
    combustor.turbine_inlet_temperature = 1450
    combustor.pressure_ratio            = 0.95
    combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()    
    
    #add the combustor to the network    
    gt_engine.combustor = combustor

    
    
    #Component 7 :core nozzle
    core_nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    core_nozzle.tag = 'core nozzle'
    
    core_nozzle.polytropic_efficiency = 0.95
    core_nozzle.pressure_ratio        = 0.99    
    
    #add the core nozzle to the network    
    gt_engine.core_nozzle = core_nozzle


    #Component 8 :fan nozzle
    fan_nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    fan_nozzle.tag = 'fan nozzle'

    fan_nozzle.polytropic_efficiency = 0.95
    fan_nozzle.pressure_ratio        = 0.99    
    
    #add the fan nozzle to the network
    gt_engine.fan_nozzle = fan_nozzle


    
    #Component 9 : fan   
    fan = SUAVE.Components.Energy.Converters.Fan()   
    fan.tag = 'fan'

    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio        = 1.7    
    
    #add the fan to the network
    gt_engine.fan = fan

    
    
    #Component 10 : thrust (to compute the thrust)
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='compute_thrust'
    
    thrust.bypass_ratio                       = 5.4
    thrust.compressor_nondimensional_massflow = 49.7272495725 #1.0
    thrust.reference_temperature              = 288.15
    thrust.reference_pressure                 = 1.01325*10**5
    thrust.design = gt_engine.thrust_design
    thrust.number_of_engines                  =gt_engine.number_of_engines   

    
    # add thrust to the network
    gt_engine.thrust = thrust



    # add  gas turbine network gt_engine to the vehicle
    vehicle.append_component(gt_engine)     
    

    ##initialize the gas turbine network
    #gt_engine                   = SUAVE.Components.Energy.Networks.Turbofan_Network()
    #gt_engine.tag               = 'Turbo Fan'
    
    #gt_engine.number_of_engines = 2.0
    #gt_engine.thrust_design     = 18800. * Units.lbf
    #gt_engine.engine_length     = 122. * Units.inches
    #gt_engine.nacelle_diameter  = 1.1 * 77. * Units.inches

    ##set the working fluid for the network
    #working_fluid               = SUAVE.Attributes.Gases.Air
    
    ##add working fluid to the network
    #gt_engine.working_fluid = working_fluid
    
    ##Component 1 : ram,  to convert freestream static to stagnation quantities
    #ram = SUAVE.Components.Energy.Converters.Ram()
    #ram.tag = 'ram'
    
    ##add ram to the network
    #gt_engine.ram = ram
    
    ##Component 2 : inlet nozzle
    #inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    #inlet_nozzle.tag = 'inlet nozzle'

    #inlet_nozzle.polytropic_efficiency = 0.99
    #inlet_nozzle.pressure_ratio        = 0.99
    
    ##add inlet nozzle to the network
    #gt_engine.inlet_nozzle = inlet_nozzle
    
    ##Component 3 :low pressure compressor    
    #low_pressure_compressor = SUAVE.Components.Energy.Converters.Compressor()    
    #low_pressure_compressor.tag = 'lpc'
    
    #low_pressure_compressor.polytropic_efficiency = 0.95
    #low_pressure_compressor.pressure_ratio        = 1.2
    
    ##add low pressure compressor to the network    
    #gt_engine.low_pressure_compressor = low_pressure_compressor
    

    ##Component 4 :high pressure compressor  
    #high_pressure_compressor = SUAVE.Components.Energy.Converters.Compressor()    
    #high_pressure_compressor.tag = 'hpc'
    
    #fpr = 1.35    # Fan Pressure Ratio (defined later)
    #opr = 59      # Overall Pressure Ratio
    #high_pressure_compressor.polytropic_efficiency = 0.95
    #high_pressure_compressor.pressure_ratio        = opr / fpr / low_pressure_compressor.pressure_ratio  
    
    ##add the high pressure compressor to the network    
    #gt_engine.high_pressure_compressor = high_pressure_compressor

    ##Component 5 :low pressure turbine  
    #low_pressure_turbine = SUAVE.Components.Energy.Converters.Turbine()   
    #low_pressure_turbine.tag='lpt'
    
    #low_pressure_turbine.mechanical_efficiency = 0.99
    #low_pressure_turbine.polytropic_efficiency = 0.95    
    
    ##add low pressure turbine to the network    
    #gt_engine.low_pressure_turbine = low_pressure_turbine
      
    
    
    ##Component 5 :high pressure turbine  
    #high_pressure_turbine = SUAVE.Components.Energy.Converters.Turbine()   
    #high_pressure_turbine.tag='hpt'

    #high_pressure_turbine.mechanical_efficiency = 0.99
    #high_pressure_turbine.polytropic_efficiency = 0.95     
    
    ##add the high pressure turbine to the network    
    #gt_engine.high_pressure_turbine = high_pressure_turbine 
    
    ##Component 6 :combustor  
    #combustor = SUAVE.Components.Energy.Converters.Combustor()   
    #combustor.tag = 'Comb'
    
    #combustor.efficiency                = 0.99 
    #combustor.alphac                    = 1.0     
    #combustor.turbine_inlet_temperature = 1500
    #combustor.pressure_ratio            = 0.95
    #combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()    
    
    ##add the combustor to the network    
    #gt_engine.combustor = combustor
    
    ##Component 7 :core nozzle
    #core_nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    #core_nozzle.tag = 'core nozzle'
    
    #core_nozzle.polytropic_efficiency = 0.95
    #core_nozzle.pressure_ratio        = 0.99    
    
    ##add the core nozzle to the network    
    #gt_engine.core_nozzle = core_nozzle


    ##Component 8 :fan nozzle
    #fan_nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    #fan_nozzle.tag = 'fan nozzle'

    #fan_nozzle.polytropic_efficiency = 0.95
    #fan_nozzle.pressure_ratio        = 0.99    
    
    ##add the fan nozzle to the network
    #gt_engine.fan_nozzle = fan_nozzle

    
    ##Component 9 : fan   
    #fan = SUAVE.Components.Energy.Converters.Fan()   
    #fan.tag = 'fan'

    #fan.polytropic_efficiency = 0.93
    #fan.pressure_ratio        = fpr
    
    ##add the fan to the network
    #gt_engine.fan = fan  
    
    ##Component 10 : thrust (to compute the thrust)
    #thrust = SUAVE.Components.Energy.Processes.Thrust()       
    #thrust.tag ='compute_thrust'
    
    #thrust.bypass_ratio                       = 13.
    #thrust.compressor_nondimensional_massflow = 49.7272495725 #Keeping 737 value...
    #thrust.reference_temperature              = 288.15
    #thrust.reference_pressure                 = 1.01325*10**5
    #thrust.design = 18800. * Units.lbf
    #thrust.number_of_engines                  =gt_engine.number_of_engines   

    
    ## add thrust to the network
    #gt_engine.thrust = thrust



    ## add  gas turbine network gt_engine to the vehicle
    #vehicle.append_component(gt_engine)     
    
    
    # ------------------------------------------------------------------
    #   Simple Aerodynamics Model
    # ------------------------------------------------------------------ 
    
    #aerodynamics = SUAVE.Attributes.Aerodynamics.Fidelity_Zero()
    aerodynamics = SUAVE.Attributes.Aerodynamics.AVL_Surrogate()
    aerodynamics.initialize(vehicle)
    
    # build stability model
    stability = SUAVE.Attributes.Flight_Dynamics.Fidelity_Zero()
    stability.initialize(vehicle)
    aerodynamics.stability = stability
    vehicle.aerodynamics_model = aerodynamics
    vehicle.aerodynamics_model.configuration.trim_drag_correction_factor = 1.0
    
    # ------------------------------------------------------------------
    #   Simple Propulsion Model
    # ------------------------------------------------------------------     
    
    vehicle.propulsion_model = gt_engine 

    # ------------------------------------------------------------------
    #   Define Configurations
    # ------------------------------------------------------------------

    # --- Takeoff Configuration ---
    config = vehicle.new_configuration("takeoff")
    # this configuration is derived from the baseline vehicle

    # --- Cruise Configuration ---
    config = vehicle.new_configuration("cruise")
    # this configuration is derived from vehicle.configs.takeoff

    # --- Takeoff Configuration ---
    takeoff_config = vehicle.configs.takeoff
    
    takeoff_config.wings['Main Wing'].flaps_angle = 20. * Units.deg
    takeoff_config.wings['Main Wing'].slats_angle = 25. * Units.deg
    
    takeoff_config.V2_VS_ratio = 1.21
    takeoff_config.maximum_lift_coefficient = 2.
    #takeoff_config.max_lift_coefficient_factor = 1.0

    # --- Landing Configuration ---
    landing_config = vehicle.new_configuration("landing")
    
    landing_config.wings['Main Wing'].flaps_angle = 30. * Units.deg
    landing_config.wings['Main Wing'].slats_angle = 25. * Units.deg

    landing_config.Vref_VS_ratio = 1.23
    landing_config.maximum_lift_coefficient = 2.
    #landing_config.max_lift_coefficient_factor = 1.0
    
    landing_config.mass_properties.landing = 0.85 * vehicle.mass_properties.takeoff
    
    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------
    
    return vehicle    


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------

def mission_setup(vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Attributes.Missions.Mission()
    mission.tag = 'The Test Mission'

    # atmospheric model
    planet = SUAVE.Attributes.Planets.Earth()
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    
    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    
    mission.airport = airport
    
    
    # ------------------------------------------------------------------
    #   First Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 1"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.takeoff
    
    # define segment attributes
    segment.atmosphere     = atmosphere
    segment.planet         = planet    
    
    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 3.0   * Units.km
    segment.air_speed      = 125.0 * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']
    
    # add to misison
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------
    #   Second Climb Segment: constant Speed, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    #segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Mach_Constant_Rate()
    segment.tag = "Climb - 2"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere     = atmosphere
    segment.planet         = planet    
    
    #segment.altitude_start = 3.0   * Units.km ## Optional
    segment.altitude_end   = 8.0   * Units.km
    segment.air_speed      = 190.0 * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']
    #segment.mach_number    = 0.5
    #segment.climb_rate     = 6.0   * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)

    
    # ------------------------------------------------------------------
    #   Third Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "Climb - 3"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet        
    
    segment.altitude_end = 35000. * Units['feet']
    segment.air_speed    = 226.0  * Units['m/s']
    segment.climb_rate   = 3.0    * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Cruise.Constant_Mach_Constant_Altitude()
    segment.tag = "Cruise"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet        
    
    segment.altitude   = 35000. * Units['feet']
    segment.mach       = 0.74
    segment.distance   = 3933.65 * Units.km
        
    mission.append_segment(segment)

    # ------------------------------------------------------------------    
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "Descent - 1"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet   
    
    segment.altitude_end = 5.0   * Units.km
    segment.air_speed    = 170.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "Descent - 2"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet    
    
    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 145.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # append to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Mission definition complete    
    # ------------------------------------------------------------------
    
    return mission


def check_results(new_results,old_results):
    
    # check segment values
    check_list = [
        'mission_profile.segments.Cruise.conditions.aerodynamics.angle_of_attack',
        'mission_profile.segments.Cruise.conditions.aerodynamics.drag_coefficient',
        'mission_profile.segments.Cruise.conditions.aerodynamics.lift_coefficient',
        'mission_profile.segments.Cruise.conditions.aerodynamics.cm_alpha',
        'mission_profile.segments.Cruise.conditions.aerodynamics.cn_beta',
        'mission_profile.segments.Cruise.conditions.propulsion.throttle',
        'mission_profile.segments.Cruise.conditions.propulsion.fuel_mass_rate',
    ]
    
    # gets a key recursively from a '.' string
    def get_key(data,keys):
        if isinstance(keys,str):
            keys = keys.split('.')
        k = keys.pop(0)
        if keys:
            return get_key(data[k],keys)
        else:
            return data[k]    
    
    # do the check
    for k in check_list:
        print k
        
        old_val = np.max( get_key(old_results,k) )
        new_val = np.max( get_key(new_results,k) )
        err = (new_val-old_val)/old_val
        print 'Error at Max:' , err
        assert np.abs(err) < 1e-6 , 'Max Check Failed : %s' % k
        
        old_val = np.min( get_key(old_results,k) )
        new_val = np.min( get_key(new_results,k) )
        err = (new_val-old_val)/old_val
        print 'Error at Min:' , err
        assert np.abs(err) < 1e-6 , 'Min Check Failed : %s' % k        
        
        print ''
    
    # check high level outputs
    def check_vals(a,b):
        if isinstance(a,Data):
            for k in a.keys():
                err = check_vals(a[k],b[k])
                if err is None: continue
                print 'outputs' , k
                print 'Error:' , err
                print ''
                assert np.abs(err) < 1e-6 , 'Outputs Check Failed : %s' % k  
        else:
            return (a-b)/a

    # do the check
    check_vals(old_results.output,new_results.output)
    
    return

    
def load_results():
    return SUAVE.Plugins.VyPy.data.load('../regresion/results_mission_B737.pkl')
    
def save_results(results):
    SUAVE.Plugins.VyPy.data.save(results,'results_mission_B737.pkl')
    
if __name__ == '__main__': 
    main()    
    plt.show()