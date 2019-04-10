# test_AVL.py
# 
# Created:  May 2017, M. Clarke
#
""" setup file for a mission with a 737 using AVL
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
Data, Container,
)

import sys

sys.path.append('../Vehicles')
# the analysis functions

from Boeing_737 import vehicle_setup, configs_setup


sys.path.append('../B737')
# the analysis functions


from mission_B737 import vehicle_setup, configs_setup, analyses_setup, mission_setup, missions_setup, simple_sizing
import copy

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main(): 
   
    # vehicle data
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # append AVL aerodynamic analysis
    aerodynamics                                               = SUAVE.Analyses.Aerodynamics.AVL()
    aerodynamics.process.compute.lift.inviscid.regression_flag = True
    aerodynamics.process.compute.lift.inviscid.keep_files      = True
    aerodynamics.geometry                                      = copy.deepcopy(configs.cruise) 
    aerodynamics.process.compute.lift.inviscid.training_file   = 'cruise_data_aerodynamics.txt'    
    configs_analyses.cruise.append(aerodynamics)     
    
    # append AVL stability analysis
    stability                                                  = SUAVE.Analyses.Stability.AVL()
    stability.regression_flag                                  = True
    stability.keep_files                                       = True
    stability.geometry                                         = copy.deepcopy(configs.cruise)
    stability.training_file                                    = 'cruise_data_stability.txt'    
    configs_analyses.cruise.append(stability)

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


    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend( configs_analyses.cruise )

    segment.air_speed = 230. * Units['m/s']
    segment.distance  = 4000. * Units.km
    segment.altitude  = 10.668 * Units.km
    
    segment.state.numerics.number_control_points = 4

    # add to mission
    mission.append_segment(segment)


    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses
    
    simple_sizing(configs, analyses)

    configs.finalize()
    analyses.finalize()
 
    # mission analysis
    mission = analyses.missions.base    
    results = mission.evaluate()

    # lift coefficient check
    lift_coefficient              = results.segments.cruise.conditions.aerodynamics.lift_coefficient[0][0]
    lift_coefficient_true         = 0.6118979131570086

    print(lift_coefficient)
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-6
    
    # moment coefficient check
    moment_coefficient            = results.segments.cruise.conditions.stability.static.CM[0][0]
    moment_coefficient_true       = -0.6267104237340391

    print(moment_coefficient)
    diff_CM                       = np.abs(moment_coefficient - moment_coefficient_true)
    print('CM difference')
    print(diff_CM)
    assert np.abs((moment_coefficient - moment_coefficient_true)/moment_coefficient_true) < 1e-6    
 
    return

if __name__ == '__main__': 
    main()    
