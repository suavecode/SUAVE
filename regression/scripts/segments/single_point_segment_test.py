# single_point_segment_test.py
# 
# Created: Feb 2020, M. Clarke
#
""" setup file for segment test regression with a Boeing 737"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Core import (
Data, Container,
)

from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Methods.Center_of_Gravity.compute_component_centers_of_gravity import compute_component_centers_of_gravity
from SUAVE.Methods.Center_of_Gravity.compute_aircraft_center_of_gravity import compute_aircraft_center_of_gravity

import sys

sys.path.append('../Vehicles')
# the analysis functions

from Boeing_737 import vehicle_setup, configs_setup
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():          
    # -----------------------------------------
    # Single Point Mission Setup 
    # -----------------------------------------
    configs, analyses = full_setup() 
    simple_sizing(configs) 
    configs.finalize()
    analyses.finalize()   
    mission = analyses.missions.base 
    results = mission.evaluate()  
    
    # check results 
    cruise_CL_1        = results.segments.single_point_1.conditions.aerodynamics.lift_coefficient[0][0]
    cruise_CL_2        = results.segments.single_point_2.conditions.aerodynamics.lift_coefficient[0][0] 
     
    cruise_CL_1_truth        = 0.2720699451586659
    cruise_CL_2_truth        = 0.2718511265980948  
    
    # Store errors 
    error_2 = Data()  
    error_2.cruise_CL_1        = np.max(np.abs(cruise_CL_1          - cruise_CL_1_truth ))     
    error_2.cruise_CL_2        = np.max(np.abs(cruise_CL_2          - cruise_CL_2_truth ))       
    
    print('Errors:')
    print(error_2)
    
    for k,v in list(error_2.items()):
        assert(np.abs(v)<1e-6) 

    
    plt.show()    
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
    mission  = mission_setup(configs_analyses)
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses
 
# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):

    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in list(configs.items()):
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
    weights = SUAVE.Analyses.Weights.Weights_Tube_Wing()
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
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.propulsors #what is called throughout the mission (at every time step))
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


def simple_sizing(configs):

    base = configs.base
    base.pull_base()

    # zero fuel weight
    base.mass_properties.max_zero_fuel = 0.9 * base.mass_properties.max_takeoff 

    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 2.0 * wing.areas.reference
        wing.areas.exposed  = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted
 
    # diff the new data
    base.store_diff() 
    
    # done!
    return 


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
#   Define the Mission
# ----------------------------------------------------------------------

def mission_setup(analyses): 

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'the_mission'

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment() 
    ones_row     = base_segment.state.ones_row     
    
    # ------------------------------------------------------------------
    #  Single Point Segment 1: constant Speed, constant altitude
    # ------------------------------------------------------------------ 
    segment = Segments.Single_Point.Set_Speed_Set_Altitude(base_segment)
    segment.tag = "single_point_1" 
    segment.analyses.extend(analyses.base) 
    segment.altitude    =  2500. * Units.feet
    segment.air_speed   =  200. * Units['m/s']  

    # add to misison
    mission.append_segment(segment)    
    
    # ------------------------------------------------------------------
    #  Single Point Segment 1: constant Speed, constant altitude
    # ------------------------------------------------------------------ 
    segment = Segments.Single_Point.Set_Speed_Set_Throttle(base_segment)
    segment.tag = "single_point_2" 
    segment.analyses.extend(analyses.base) 
    segment.altitude    =  2500. * Units.feet
    segment.air_speed   =  200. * Units['m/s']   
    segment.throttle    =  0.6

    # add to misison
    mission.append_segment(segment)         
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

if __name__ == '__main__': 
    main()    
