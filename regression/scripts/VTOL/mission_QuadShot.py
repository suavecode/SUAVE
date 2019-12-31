# vtol_UAV.py
# 
# Created:  Jan 2016, E. Botero
# Modified: 

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units, Data

import numpy as np
import pylab as plt
import matplotlib
import copy, time
from SUAVE.Plots.Mission_Plots import *
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power, initialize_from_mass
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_kv
import cProfile, pstats, io
#from SUAVE.Components.Energy.Processes.propeller_map import propeller_map
import sys

sys.path.append('../Vehicles')
# the analysis functions

from QuadShot import vehicle_setup, configs_setup

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    # build the vehicle, configs, and analyses
    configs, analyses = full_setup()
    
    configs.finalize()
    analyses.finalize()    
    
    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()
    
    #eva = mission.evaluate
    
    #pr = cProfile.Profile(timeunit=10)
    #pr.runctx('eva()',None,locals())
    
    #s = io.StringIO()
    #sortby = 'tottime'
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print s.getvalue()        
        
    # plot results    
    plot_mission(results)
    
    return

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():
    
    # vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)
    
    # vehicle analyses
    configs_analyses = analyses_setup(configs)
    
    # mission analyses
    mission  = mission_setup(configs_analyses,vehicle)
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses
    
    return configs, analyses


def configs_setup(vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------
    
    configs = SUAVE.Components.Configs.Config.Container()
    
    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)
    
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
    weights.settings.empty_weight_method = \
        SUAVE.Methods.Weights.Correlations.UAV.empty
    weights.vehicle = vehicle
    analyses.append(weights)
    
    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.AERODAS()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    aerodynamics.settings.maximum_lift_coefficient   = 1.5
    analyses.append(aerodynamics)    
    
    # ------------------------------------------------------------------
    #  Energy
    energy = SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.propulsors
    analyses.append(energy)
    
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
#   Define the Mission
# ----------------------------------------------------------------------
def mission_setup(analyses,vehicle):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'The Test Mission'

    mission.atmosphere  = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    mission.planet      = SUAVE.Attributes.Planets.Earth()
    
    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments
    
    # base segment
    base_segment = Segments.Segment()   
    ones_row     = base_segment.state.ones_row
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals
    base_segment.state.unknowns.propeller_power_coefficient  = 0.001 * ones_row(1) 
    base_segment.state.unknowns.battery_voltage_under_load   = vehicle.propulsors.propulsor.battery.max_voltage * ones_row(1) 
    base_segment.state.residuals.network                     = 0. * ones_row(2) 

    #------------------------------------------------------------------    
    #  Climb Hover
    #------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Mission.Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "Climb"
    
    # connect vehicle configuration
    segment.analyses.extend(analyses.base)
    
    # segment attributes   
    ones_row = segment.state.ones_row
    segment.battery_energy  = vehicle.propulsors.propulsor.battery.max_energy
    segment.altitude_start  = 0.
    segment.altitude_end    = 100. * Units.m
    segment.climb_rate      = 3.  * Units.m / Units.s 
    segment.air_speed       = 3.  * Units.m / Units.s
    segment.state.unknowns.body_angle  = ones_row(1) * 90. *Units.deg
    
    mission.append_segment(segment)   
    
    # ------------------------------------------------------------------    
    #   Hover
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Mission.Segments.Hover.Hover(base_segment)
    segment.tag = "Hover_1"
    
    # connect vehicle configuration
    segment.analyses.extend(analyses.base)
    
    # segment attributes     
    segment.time = 60* Units.seconds
    segment.state.conditions.frames.body.inertial_rotations[:,1] = ones_row(1) * 90.*Units.deg 
    
    mission.append_segment(segment)    
    
    # ------------------------------------------------------------------    
    #   Hover Transition
    # ------------------------------------------------------------------     
    
    segment = SUAVE.Analyses.Mission.Segments.Cruise.Constant_Acceleration_Constant_Altitude(base_segment)
    segment.tag = "Transition_to_Cruise"
    
    # connect vehicle configuration
    segment.analyses.extend(analyses.base)
    
    # segment attributes       
    segment.acceleration      = 1.5 * Units['m/s/s']
    segment.air_speed_initial = 0.0
    segment.air_speed_final   = 15.0 
    segment.altitude          = 100. * Units.m
    
    mission.append_segment(segment)      
           

    # ------------------------------------------------------------------    
    #   Cruise
    # ------------------------------------------------------------------     
    
    segment = SUAVE.Analyses.Mission.Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "Cruise"
    
    # connect vehicle configuration
    segment.analyses.extend(analyses.base)
    
    # segment attributes     
    segment.distance  = 2.  * Units.km
    segment.air_speed = 15. * Units.m/Units.s

    mission.append_segment(segment)            
    
    # ------------------------------------------------------------------    
    #   Hover Transition
    # ------------------------------------------------------------------     
    
    segment = SUAVE.Analyses.Mission.Segments.Cruise.Constant_Acceleration_Constant_Altitude(base_segment)
    segment.tag = "Transition_to_hover"
    
    # connect vehicle configuration
    segment.analyses.extend(analyses.base)
    
    # segment attributes       
    segment.acceleration      = -0.5 * Units['m/s/s']
    segment.air_speed_initial = 15.0
    segment.air_speed_final   = 0.0 
    segment.altitude = 100. * Units.m
    
    mission.append_segment(segment)  
    
    # ------------------------------------------------------------------    
    #   Hover
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Mission.Segments.Hover.Hover(base_segment)
    segment.tag = "Hover_2"
    
    # connect vehicle configuration
    segment.analyses.extend(analyses.base)
    
    # segment attributes     
    segment.time = 60* Units.seconds
    segment.state.conditions.frames.body.inertial_rotations[:,1] = ones_row(1)* 90.*Units.deg
    
    mission.append_segment(segment)        
    
    # ------------------------------------------------------------------    
    #   Descent Hover
    # ------------------------------------------------------------------    
    
    segment = SUAVE.Analyses.Mission.Segments.Hover.Descent(base_segment)
    segment.tag = "Descent"
    
    # connect vehicle configuration
    segment.analyses.extend(analyses.base)
    
    # segment attributes     
    segment.altitude_end = 0. * Units.m
    segment.descent_rate = 3. * Units.m / Units.s   
    segment.state.conditions.frames.body.inertial_rotations[:,1] = ones_row(1)* 90.*Units.deg
    
    mission.append_segment(segment)       
    
    # ------------------------------------------------------------------    
    #   Mission definition complete    
    #------------------------------------------------------------------
    
    return mission

def missions_setup(base_mission):

    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()
    
    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------
    
    missions.base = base_mission
    
    # done!
    return missions  

# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def plot_mission(results):

    plot_aircraft_velocities(results)
    
    plot_aerodynamic_coefficients(results)
    
    plot_aerodynamic_forces(results) 
    
    plot_electronic_conditions(results)
    
    plot_flight_conditions(results) 
    
    plot_lift_cruise_network(results) 
    
    return     

if __name__ == '__main__':
    main()
    
    plt.show()