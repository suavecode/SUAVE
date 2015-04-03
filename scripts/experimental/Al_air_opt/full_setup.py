# full_setup.py
# 
# Created:  SUave Team, Aug 2014
# Modified: 

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
Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#   Vehicle Setup
# ----------------------------------------------------------------------

def full_setup():
    
    from vehicle_and_mission_setup import vehicle_setup,mission_setup

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
#   Define the Configurations
# ----------------------------------------------------------------------

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
    
    configs.append(config)
   
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'landing'
    
    config.wings['main_wing'].flaps.angle = 30. * Units.deg
    config.wings['main_wing'].slats.angle = 25. * Units.deg
    config.Vref_VS_ratio = 1.23
    
    configs.append(config)    
    
    # done!
    return configs

# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):
    
    analyses = SUAVE.Analyses.Analysis.Container()
    
    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis
        
    # adjust analyses for configs
    
    # takeoff_analysis
    analyses.takeoff.aerodynamics.drag_coefficient_increment = 0.1000
    
    # landing analysis
    aerodynamics = analyses.landing.aerodynamics
    # do something here eventually
    
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
    weights.settings.empty_weight_method= \
    SUAVE.Methods.Weights.Correlations.Tube_Wing.empty_custom_eng
    weights.vehicle = vehicle
    analyses.append(weights)
    
    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)
    
    # ------------------------------------------------------------------
    #  Stability Analysis
    stability = SUAVE.Analyses.Stability.Fidelity_Zero()
    stability.geometry = vehicle
    analyses.append(stability)
    
    # ------------------------------------------------------------------
    #  Propulsion Analysis
    propulsion = SUAVE.Analyses.Energy.Propulsion()
    propulsion.propulsor = vehicle.propulsors['network']
    analyses.append(propulsion)
    
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
#   Various Missions
# ----------------------------------------------------------------------
    
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
    fuel_mission = SUAVE.Analyses.Mission.Sequential_Segments() #Fuel_Constrained()
    fuel_mission.tag = 'fuel'
    fuel_mission.mission = base_mission
    fuel_mission.range   = 1277. * Units.nautical_mile
    fuel_mission.payload   = 19000.
    missions.append(fuel_mission)    
    
    # ------------------------------------------------------------------
    #   Mission for Constrained Short Field
    # ------------------------------------------------------------------    
    short_field = SUAVE.Analyses.Mission.Sequential_Segments() #Short_Field_Constrained()
    short_field.mission = base_mission
    short_field.tag = 'short_field'    
    
    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    airport.available_tofl = 1500.
    short_field.mission.airport = airport    
    missions.append(short_field)
   
    # ------------------------------------------------------------------
    #   Mission for Fixed Payload
    # ------------------------------------------------------------------    
    payload = SUAVE.Analyses.Mission.Sequential_Segments() #Payload_Constrained()
    payload.mission = base_mission
    payload.tag = 'payload'
    payload.range   = 2316. * Units.nautical_mile
    payload.payload   = 19000.
    missions.append(payload)
    
    
    # done!
    return missions   

# ----------------------------------------------------------------------
#   Apply Simple Sizing Principles
# ----------------------------------------------------------------------

def simple_sizing(configs, analyses, m_guess, Ereq, Preq):
    from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform
    
# ------------------------------------------------------------------
    #   Define New Gross Takeoff Weight
    # ------------------------------------------------------------------
    #now add component weights to the gross takeoff weight of the vehicle
   
    base = configs.base
    base.pull_base()
    #determine geometry of fuselage as well as wings
    fuselage=base.fuselages['fuselage']
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.fuselage_planform(fuselage)
    fuselage.areas.side_projected   = fuselage.heights.maximum*fuselage.lengths.cabin*1.1 #  Not correct
    base.wings['main_wing'] = wing_planform(base.wings['main_wing'])
    base.wings['horizontal_stabilizer'] = wing_planform(base.wings['horizontal_stabilizer']) 
    base.wings['vertical_stabilizer']   = wing_planform(base.wings['vertical_stabilizer'])   
    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 2.00 * wing.areas.reference
        wing.areas.affected = 0.60 * wing.areas.reference
        wing.areas.exposed  = 0.75 * wing.areas.wetted
  
    battery=base.propulsors.network['battery']
    ducted_fan=base.propulsors.network['ducted_fan']
    SUAVE.Methods.Power.Battery.Sizing.initialize_from_energy_and_power(battery,Ereq,Preq)
    battery.current_energy=[battery.max_energy] #initialize list of current energy
    m_air       =SUAVE.Methods.Power.Battery.Variable_Mass.find_total_mass_gain(battery)
    m_water     =battery.find_water_mass()
    #now add the electric motor weight
    motor_mass=ducted_fan.number_of_engines*SUAVE.Methods.Weights.Correlations.Propulsion.air_cooled_motor((Preq)*Units.watts/ducted_fan.number_of_engines)
    propulsion_mass=SUAVE.Methods.Weights.Correlations.Propulsion.integrated_propulsion(motor_mass/ducted_fan.number_of_engines,ducted_fan.number_of_engines)
    
    ducted_fan.mass_properties.mass=propulsion_mass
   
    breakdown = analyses.configs.base.weights.evaluate()
    breakdown.battery=battery.mass_properties.mass
    
    base.mass_properties.breakdown=breakdown
    m_fuel=0.
    
    base.mass_properties.operating_empty     = breakdown.empty 
    
    #weight =SUAVE.Methods.Weights.Correlations.Tube_Wing.empty_custom_eng(vehicle, ducted_fan)
    m_full=breakdown.empty+battery.mass_properties.mass+breakdown.payload
    m_end=m_full+m_air
    base.mass_properties.takeoff                 = m_full
    base.store_diff()

    # Update all configs with new base data    
    for config in configs:
        config.pull_base()

    
    ##############################################################################
    # ------------------------------------------------------------------
    #   Define Configurations
    # ------------------------------------------------------------------
    '''
    takeoff_config=configs.takeoff
    takeoff_config.pull_base()
    takeoff_config.mass_properties.takeoff= m_full
    takeoff_config.store_diff()
    '''
    landing_config=configs.landing
    
    landing_config.wings['main_wing'].flaps.angle =  50. * Units.deg
    landing_config.wings['main_wing'].slats.angle  = 25. * Units.deg
    landing_config.mass_properties.landing = m_end
    landing_config.store_diff()
        
    
    #analyses.weights=configs.base.mass_properties.takeoff
    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------
    
    return 

if __name__ == '__main__':
    full_setup()  