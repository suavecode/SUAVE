# full_setup_737800.py
# 
# Created:  SUave Team    , Aug 2014
# Modified: Tim MacDonald , Sep 2014

""" setup file for a mission with a Boeing 737-800 single aisle
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

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components import Component_Exception
from SUAVE.Components.Energy.Gas_Turbine import Network


def full_setup_737800_gasturbine_network():

    vehicle = vehicle_setup()
    mission = mission_setup(vehicle)
    
    return vehicle, mission

def vehicle_setup():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Boeing 737-800'    
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff               = 79010.0   # kg
    vehicle.mass_properties.operating_empty           = 41145.0   # kg
    vehicle.mass_properties.takeoff                   = 79015.8   # kg
    vehicle.mass_properties.max_zero_fuel             = 0.9 * vehicle.mass_properties.max_takeoff 
    vehicle.mass_properties.cargo                     = 10000.  * Units.kilogram   
    
    vehicle.mass_properties.center_of_gravity         = [60 * Units.feet, 0, 0]  # Not correct
    vehicle.mass_properties.moments_of_inertia.tensor = [[10 ** 5, 0, 0],[0, 10 ** 6, 0,],[0,0, 10 ** 7]] # Not Correct
    
    # envelope properties
    vehicle.envelope.ultimate_load = 3.5
    vehicle.envelope.limit_load    = 1.5

    # basic parameters
    vehicle.reference_area        = 125.0      
    vehicle.passengers = 160
    vehicle.systems.control  = "fully powered" 
    vehicle.systems.accessories = "medium range"
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Main Wing'
    
    wing.areas.reference = 125.0      #
    wing.aspect_ratio    = 9.45       #
    wing.spans.projected = 35.7       #
    wing.sweep           = 25 * Units.deg
    wing.symmetric       = True
    wing.thickness_to_chord = 0.1
    wing.taper           = 0.16
    
    
    # size the wing planform ----------------------------------
    # These can be determined by the wing sizing function
    # Note that the wing sizing function will overwrite span
    wing.chords.root  = 6.81
    wing.chords.tip   = 1.09
    wing.areas.wetted = wing.areas.reference*2.0 
    # The span that would normally be overwritten here doesn't match
    # ---------------------------------------------------------
    
    wing.chords.mean_aerodynamic = 12.5
    wing.areas.exposed = 0.8*wing.areas.wetted
    wing.areas.affected = 0.6*wing.areas.wetted
    wing.span_efficiency = 0.9
    wing.twists.root = 3.0*Units.degrees
    wing.twists.tip  = 3.0*Units.degrees
    wing.origin          = [20,0,0]
    wing.aerodynamic_center = [3,0,0] 
    wing.vertical   = False
    wing.eta         = 1.0
    
    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Horizontal Stabilizer'
    
    wing.areas.reference = 32.78     #
    wing.aspect_ratio    = 6.16      #
    wing.spans.projected = 14.35     #
    wing.sweep           = 30 * Units.deg
    wing.symmetric       = True
    wing.thickness_to_chord = 0.08
    wing.taper           = 0.2
    
    # size the wing planform ----------------------------------
    # These can be determined by the wing sizing function
    # Note that the wing sizing function will overwrite span
    wing.chords.root  = 3.28
    wing.chords.tip   = 1.31
    wing.areas.wetted = wing.areas.reference*2.0 
    # ---------------------------------------------------------
    
    wing.chords.mean_aerodynamic = 8.0
    wing.areas.exposed = 0.8*wing.areas.wetted
    wing.areas.affected = 0.6*wing.areas.wetted
    wing.span_efficiency = 0.9
    wing.twists.root = 3.0*Units.degrees
    wing.twists.tip  = 3.0*Units.degrees  
    wing.origin          = [50,0,0]
    wing.aerodynamic_center = [2,0,0]
    wing.vertical   = False 
    wing.eta         = 0.9  
    
    # add to vehicle
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'Vertical Stabilizer'    
    
    wing.areas.reference = 26.44     #
    wing.aspect_ratio    = 1.91      #
    wing.spans.projected = 7.877     #
    wing.sweep           = 35 * Units.deg
    wing.symmetric       = False
    wing.thickness_to_chord = 0.08
    wing.taper           = 0.27
    
    # size the wing planform ----------------------------------
    # These can be determined by the wing sizing function
    # Note that the wing sizing function will overwrite span
    wing.chords.root  = 6.60
    wing.chords.tip   = 1.65
    wing.areas.wetted = wing.areas.reference*2.0 
    # ---------------------------------------------------------
    
    wing.chords.mean_aerodynamic = 8.0
    wing.areas.exposed = 1.0*wing.areas.wetted
    wing.areas.affected = 0.6*wing.areas.wetted
    wing.span_efficiency = 0.9
    wing.twists.root = 0.0*Units.degrees
    wing.twists.tip  = 0.0*Units.degrees  
    wing.origin          = [50,0,0]
    wing.aerodynamic_center = [2,0,0]    
    wing.vertical   = True 
    wing.t_tail     = False
    wing.eta         = 1.0
        
    # add to vehicle
    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'Fuselage'
    
    fuselage.number_coach_seats = 160
    fuselage.seats_abreast = 6
    fuselage.seat_pitch = 1
    fuselage.fineness.nose = 1.6
    fuselage.fineness.tail = 2.
    fuselage.lengths.fore_space = 6.
    fuselage.lengths.aft_space  = 5.
    fuselage.width = 4.
    fuselage.heights.maximum          = 4.    #
    fuselage.areas.side_projected       = 4.* 59.8 #  Not correct
    fuselage.heights.at_quarter_length = 4. 
    fuselage.heights.at_three_quarters_length = 4. 
    fuselage.heights.at_wing_root_quarter_chord = 4. 
    fuselage.differential_pressure = 10**5   * Units.pascal    # Maximum differential pressure
    
    # size fuselage planform
    # A function exists to do this
    fuselage.lengths.nose  = 6.4
    fuselage.lengths.tail  = 8.0
    fuselage.lengths.cabin = 44.0
    fuselage.lengths.total = 58.4
    fuselage.areas.wetted  = 688.64
    fuselage.areas.front_projected = 12.57
    fuselage.effective_diameter        = 4.0
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
    # ------------------------------------------------------------------
    #  Turbofan
    # ------------------------------------------------------------------    
    
    turbofan = SUAVE.Components.Propulsors.TurboFanPASS()
    turbofan.tag = 'Turbo Fan'
    
    turbofan.propellant = SUAVE.Attributes.Propellants.Jet_A()
    
    #turbofan.analysis_type                 = '1D'     #
    turbofan.diffuser_pressure_ratio       = 0.98     #
    turbofan.fan_pressure_ratio            = 1.7      #
    turbofan.fan_nozzle_pressure_ratio     = 0.99     #
    turbofan.lpc_pressure_ratio            = 1.14     #
    turbofan.hpc_pressure_ratio            = 13.415   #
    turbofan.burner_pressure_ratio         = 0.95     #
    turbofan.turbine_nozzle_pressure_ratio = 0.99     #
    turbofan.Tt4                           = 1450.0   #
    turbofan.bypass_ratio                  = 5.4      #
    turbofan.thrust.design                 = 25000.0  #
    turbofan.number_of_engines             = 2.0      #
    
 #   turbofan.lengths.engine                = 3.0
    
    # size the turbofan
    turbofan.A2          =   1.753
    turbofan.df          =   1.580
    turbofan.nacelle_dia =   1.580
    turbofan.A2_5        =   0.553
    turbofan.dhc         =   0.857
    turbofan.A7          =   0.801
    turbofan.A5          =   0.191
    turbofan.Ao          =   1.506
    turbofan.mdt         =   9.51
    turbofan.mlt         =  22.29
    turbofan.mdf         = 355.4
    turbofan.mdlc        =  55.53
    turbofan.D           =   1.494
    turbofan.mdhc        =  49.73  
    
    # add to vehicle
    #vehicle.append_component(turbofan)    

    
    # ------------------------------------------------------------------
    #  Turbofan Network
    # ------------------------------------------------------------------    
    

    #initialize the gas turbine network
    gt_engine = SUAVE.Components.Energy.Networks.Turbofan_Network()
    #gt_engine = SUAVE.Components.Energy.Gas_Turbine.Network()

    gt_engine.tag = 'Turbo Fan'
    #GT_ENGINE.WORKING_FLUID
    
    gt_engine.number_of_engines = 2.0
    gt_engine.thrust_design = 24000.0
    

    
    working_fluid = SUAVE.Attributes.Gases.Air
    gt_engine.working_fluid = working_fluid
    
    
    #create a ram component to convert the freestream quantities to stagnation quantities
    ram = SUAVE.Components.Energy.Converters.Ram()
    #ram = SUAVE.Components.Energy.Gas_Turbine.Ram()

    ram.tag = 'ram'
    gt_engine.ram = ram
    
    
    
    
    #create the inlet nozzle to the engine 
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    #inlet_nozzle = SUAVE.Components.Energy.Gas_Turbine.Compression_Nozzle()
    inlet_nozzle.tag = 'inlet nozzle'
    #gt_engine.inlet_nozzle = inlet_nozzle
    # input the pressure ratio and polytropic effeciency of the nozzle
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio = 0.98
    gt_engine.inlet_nozzle = inlet_nozzle
    
    
    #low pressure compressor    
    low_pressure_compressor = SUAVE.Components.Energy.Converters.Compressor()    
    #low_pressure_compressor = SUAVE.Components.Energy.Gas_Turbine.Compressor()
    low_pressure_compressor.tag = 'lpc'
    # input the pressure ratio and polytropic effeciency of the compressor
    low_pressure_compressor.polytropic_efficiency = 0.91
    low_pressure_compressor.pressure_ratio = 1.14    
    gt_engine.low_pressure_compressor = low_pressure_compressor

    
    

      
    #high pressure compressor  
    high_pressure_compressor = SUAVE.Components.Energy.Converters.Compressor()    
    #high_pressure_compressor = SUAVE.Components.Energy.Gas_Turbine.Compressor()
    high_pressure_compressor.tag = 'hpc'
    # input the pressure ratio and polytropic effeciency of the compressor
    high_pressure_compressor.polytropic_efficiency = 0.91
    high_pressure_compressor.pressure_ratio = 13.415    
    gt_engine.high_pressure_compressor = high_pressure_compressor

    
 

    
    #low pressure turbine  
    low_pressure_turbine = SUAVE.Components.Energy.Converters.Turbine()   
    #low_pressure_turbine = SUAVE.Components.Energy.Gas_Turbine.Turbine()
    low_pressure_turbine.tag='lpt'
    # input the pressure ratio and polytropic effeciency of the turbine
    low_pressure_turbine.mechanical_efficiency =0.99
    low_pressure_turbine.polytropic_efficiency = 0.93     
    gt_engine.low_pressure_turbine = low_pressure_turbine
      
    
    
    #high pressure turbine  
    high_pressure_turbine = SUAVE.Components.Energy.Converters.Turbine()   
    #high_pressure_turbine = SUAVE.Components.Energy.Gas_Turbine.Turbine()
    high_pressure_turbine.tag='hpt'
    # input the pressure ratio and polytropic effeciency of the turbine
    high_pressure_turbine.mechanical_efficiency =0.99
    high_pressure_turbine.polytropic_efficiency = 0.93     
    gt_engine.high_pressure_turbine = high_pressure_turbine 
      
    
    
    #combustor  
    combustor = SUAVE.Components.Energy.Converters.Combustor()   
    #combustor = SUAVE.Components.Energy.Gas_Turbine.Combustor()
    combustor.tag = 'Comb'
    # input the effeciency, pressure ratio and the turbine inlet temperature
    combustor.efficiency = 0.99 #where is this
    combustor.alphac = 1.0     
    combustor.turbine_inlet_temperature =   1450
    combustor.pressure_ratio =   0.95
    fuel_data = SUAVE.Attributes.Propellants.Jet_A()    
    gt_engine.combustor = combustor

    
    
    #core nozzle
    core_nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    #core_nozzle = SUAVE.Components.Energy.Gas_Turbine.Expansion_Nozzle()
    core_nozzle.tag = 'core nozzle'
    # input the pressure ratio and polytropic effeciency of the nozzle
    core_nozzle.polytropic_efficiency = 0.95
    core_nozzle.pressure_ratio = 0.99    
    gt_engine.core_nozzle = core_nozzle

     



    #fan nozzle
    fan_nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    #fan_nozzle = SUAVE.Components.Energy.Gas_Turbine.Expansion_Nozzle()
    fan_nozzle.tag = 'fan nozzle'
    # input the pressure ratio and polytropic effeciency of the nozzle
    fan_nozzle.polytropic_efficiency = 0.95
    fan_nozzle.pressure_ratio = 0.99    
    gt_engine.fan_nozzle = fan_nozzle


    
    #power out as an output
    #fan    
    fan = SUAVE.Components.Energy.Converters.Fan()   
    #fan = SUAVE.Components.Energy.Gas_Turbine.Fan()
    fan.tag = 'fan'
    # input the pressure ratio and polytropic effeciency of the fan
    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio = 1.7    
    gt_engine.fan = fan

    
    
    #create a thrust component which computes the thrust of the engine
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    #thrust = SUAVE.Components.Energy.Gas_Turbine.Thrust()
    thrust.tag ='compute_thrust'
    thrust.bypass_ratio=5.4
    thrust.compressor_nondimensional_massflow = 49.7272495725 #1.0
    thrust.reference_temperature =288.15
    thrust.reference_pressure=1.01325*10**5
    thrust.number_of_engines =gt_engine.number_of_engines     
    gt_engine.thrust = thrust
    gt_engine.thrust.design = 24000.0

    
    # add to vehicle
    vehicle.append_component(gt_engine)     
    
    # ------------------------------------------------------------------
    #   Simple Aerodynamics Model
    # ------------------------------------------------------------------ 
    
    aerodynamics = SUAVE.Attributes.Aerodynamics.Fidelity_Zero()
    aerodynamics.initialize(vehicle)
    
    # build stability model
    stability = SUAVE.Attributes.Flight_Dynamics.Fidelity_Zero()
    stability.initialize(vehicle)
    aerodynamics.stability = stability
    vehicle.aerodynamics_model = aerodynamics
    
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
    takeoff_config.wings['Main Wing'].flaps_angle =  20. * Units.deg
    takeoff_config.wings['Main Wing'].slats_angle  = 25. * Units.deg
    # V2_V2_ratio may be informed by user. If not, use default value (1.2)
    takeoff_config.V2_VS_ratio = 1.21
    # CLmax for a given configuration may be informed by user. If not, is calculated using correlations
    takeoff_config.maximum_lift_coefficient = 2.
    #takeoff_config.max_lift_coefficient_factor = 1.0

    # --- Landing Configuration ---
    landing_config = vehicle.new_configuration("landing")
    landing_config.wings['Main Wing'].flaps_angle =  30. * Units.deg
    landing_config.wings['Main Wing'].slats_angle  = 25. * Units.deg
    # Vref_V2_ratio may be informed by user. If not, use default value (1.23)
    landing_config.Vref_VS_ratio = 1.23
    # CLmax for a given configuration may be informed by user
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
    
    segment.altitude_end = 10.668 * Units.km
    segment.air_speed    = 226.0  * Units['m/s']
    segment.climb_rate   = 3.0    * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Attributes.Missions.Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = "Cruise"
    
    # connect vehicle configuration
    segment.config = vehicle.configs.cruise
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet        
    
    #segment.altitude   = 10.668  * Units.km     # Optional
    segment.air_speed  = 230.412 * Units['m/s']
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


if __name__ == '__main__': 
    
    full_setup_737800_gasturbine_network()