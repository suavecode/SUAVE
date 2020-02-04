# test_VTOL.py
# 
# Created:  July 2018, M. Clarke
#
""" setup file for a mission with a Stopped Rotor and Tiltwing eVTOL Common Reference Vehicles (CRMs)
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

sys.path.append('../VTOL')
# the analysis functions 
 
from mission_Tiltwing_CRM      import tw_full_setup, tw_plot_mission
from mission_Stopped_Rotor_CRM import sr_full_setup, sr_plot_mission
import copy

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main(): 
    # ------------------------------------------------------------------------------------------------------------
    # Tiltwing CRM  
    # ------------------------------------------------------------------------------------------------------------
    # build the vehicle, configs, and analyses
    configs, analyses = tw_full_setup() 
    configs.finalize()
    analyses.finalize()
    weights   = analyses.configs.base.weights
    breakdown = weights.evaluate()    
    mission   = analyses.missions.base
    
    # evaluate mission    
    tw_results   = mission.evaluate()  
    
    # plot results
    tw_plot_mission(tw_results)    
    
    # save, load and plot old results 
    #save_tiltwing_results(tw_results)
    tw_old_results = load_tiltwing_results()
    tw_plot_mission(tw_old_results)   
    
    # RPM check during hover
    RPM        = tw_results.segments.hover.conditions.propulsion.rpm[0][0]
    RPM_true   = 1018.1932516955794
    print(RPM) 
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  

    # lift Coefficient Check During Cruise
    lift_coefficient        = tw_results.segments.cruise.conditions.aerodynamics.lift_coefficient[0][0] 
    lift_coefficient_true   = 0.65095707
    print(lift_coefficient)
    diff_CL                 = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-3    
        
        
    # ------------------------------------------------------------------------------------------------------------------
    # Stopped-Rotor CRM   
    # ------------------------------------------------------------------------------------------------------------------
    # build the vehicle, configs, and analyses
    configs, analyses = sr_full_setup() 
    analyses.finalize()     
    weights   = analyses.weights
    breakdown = weights.evaluate() 
    mission   = analyses.mission  
    
    # evaluate mission     
    sr_results   = mission.evaluate()
        
    # plot results
    sr_plot_mission(sr_results,configs)
    
    # save, load and plot old results 
    #save_stopped_rotor_results(sr_results)
    sr_old_results = load_stopped_rotor_results()
    sr_plot_mission(sr_old_results,configs) 
 
    
    # RPM of rotor check during hover
    RPM        = sr_results.segments.climb_1.conditions.propulsion.rpm_lift[0][0]
    RPM_true   = 2301.9958069793593
    print(RPM) 
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  
    
    # Battery Energy Check During Transition
    battery_energy_trans_to_hover         = sr_results.segments.transition_1.conditions.propulsion.battery_energy[:,0]
    battery_energy_trans_to_hover_true    = np.array([3.06540422e+08, 3.06247811e+08, 3.05400563e+08, 3.04104017e+08,
                                                      3.02562940e+08, 3.01011637e+08, 2.99666944e+08, 2.98684953e+08,
                                                      2.98110825e+08, 2.97924248e+08])
    print(battery_energy_trans_to_hover)
    diff_battery_energy_trans_to_hover    = np.abs(battery_energy_trans_to_hover  - battery_energy_trans_to_hover_true) 
    print('battery_energy_trans_to_hover difference')
    print(diff_battery_energy_trans_to_hover)   
    assert all(np.abs((battery_energy_trans_to_hover - battery_energy_trans_to_hover_true)/battery_energy_trans_to_hover) < 1e-3)

    # lift Coefficient Check During Cruise
    lift_coefficient        = sr_results.segments.cruise.conditions.aerodynamics.lift_coefficient[0][0]
    lift_coefficient_true   = 0.6962308250135634
    print(lift_coefficient)
    diff_CL                 = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-3    
    
    return


def load_tiltwing_results():
    return SUAVE.Input_Output.SUAVE.load('results_tiltwing.res')

def save_tiltwing_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'results_tiltwing.res')
    return

def load_stopped_rotor_results():
    return SUAVE.Input_Output.SUAVE.load('results_stopped_rotor.res')

def save_stopped_rotor_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'results_stopped_rotor.res')
    return

if __name__ == '__main__': 
    main()    
