# test_VTOL.py
# 
# Created:  July 2018, M. Clarke
#
""" setup file for a mission with a QuadShot
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

from QuadShot import vehicle_setup, configs_setup


sys.path.append('../VTOL')
# the analysis functions


from mission_QuadShot import vehicle_setup, configs_setup, analyses_setup, mission_setup, missions_setup
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
 
    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics                                  = SUAVE.Analyses.Aerodynamics.AERODAS()
    aerodynamics.geometry                         = copy.deepcopy(configs.base)   
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    aerodynamics.settings.maximum_lift_coefficient   = 1.5    
    configs_analyses.base.append(aerodynamics)     
  
    # ------------------------------------------------------------------        
    # mission analyses
    mission  = mission_setup(configs_analyses,vehicle)
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    configs.finalize()
    analyses.finalize()

    # mission analysis
    mission = analyses.missions.base    
    results = mission.evaluate()

    # RPM check during hover
    RPM            = results.conditions.hover_1.propulsion.rpm[0] 
    RPM_true       = 4685.21033888
    print RPM 
    diff_RPM                        = np.abs(RPM - RPM_true)
    print 'RPM difference'
    print diff_RPM
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  
    
    # battery energy check during transition
    battery_energy_trans_to_hover              = results.conditions.transition_to_hover.propulsion.battery_energy[0]
    battery_energy_trans_to_hover_true         = 92097.82354179
    print battery_energy_trans_to_hover
    diff_battery_energy_trans_to_hover                      = np.abs(battery_energy_trans_to_hover  - battery_energy_trans_to_hover_true) 
    print 'battery_energy_trans_to_hover difference'
    print diff_battery_energy_trans_to_hover
    assert np.abs((battery_energy_trans_to_hover  - battery_energy_trans_to_hover_true)/battery_energy_trans_to_hover) < 1e-3


    # lift coefficient check during cruise
    lift_coefficient              = results.conditions.cruise.aerodynamics.lift_coefficient[0]
    lift_coefficient_true         = 0.3527293
    print lift_coefficient
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true) 
    print 'CL difference'
    print diff_CL
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-3

  

    return

if __name__ == '__main__': 
    main()    
