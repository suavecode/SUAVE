# test_VTOL.py
# 
# Created:  Feb 2020, M. Clarke
#
""" setup file for a mission with an Electic Helicopter, Stopped Roto,
 and Tiltwing eVTOL Common Reference Vehicles (CRMs)
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
 
from mission_Tiltwing_CRM        import tw_full_setup, tw_plot_mission
from mission_Stopped_Rotor_CRM   import sr_full_setup, sr_plot_mission
from mission_Electric_Helicopter import eh_full_setup, eh_plot_mission
import copy

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():  
    # ------------------------------------------------------------------------------------------------------------------
    # Electric Helicopter  
    # ------------------------------------------------------------------------------------------------------------------
    # build the vehicle, configs, and analyses 
    configs, analyses = eh_full_setup() 
    analyses.finalize()    
    weights      = analyses.configs.base.weights
    breakdown    = weights.evaluate()     
    mission      = analyses.missions.base  
    eh_results   = mission.evaluate()
        
    # plot results
    eh_plot_mission(eh_results)
    
    # save, load and plot old results 
    #save_electric_helicopter_results(eh_results)
    eh_old_results = load_electric_helicopter_results()
    eh_plot_mission(eh_old_results) 
 
    
    # RPM of rotor check during hover
    RPM        = eh_results.segments.climb.conditions.propulsion.rpm[0][0]
    RPM_true   = 1598.7692155337331
    print(RPM) 
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  
    
    # Battery Energy Check During Transition
    battery_energy_transition         = eh_results.segments.hover.conditions.propulsion.battery_energy[:,0]
    battery_energy_transition_true    = np.array([3.57097712e+08, 3.56795677e+08, 3.55902186e+08, 3.54454664e+08,
                                                  3.52514086e+08, 3.50162799e+08, 3.47501414e+08, 3.44644776e+08,
                                                  3.41717150e+08, 3.38846818e+08, 3.36160348e+08, 3.33776835e+08,
                                                  3.31802409e+08, 3.30325288e+08, 3.29411640e+08, 3.29102462e+08])
    print(battery_energy_transition)
    diff_battery_energy_transition    = np.abs(battery_energy_transition  - battery_energy_transition_true) 
    print('battery energy of transition')
    print(diff_battery_energy_transition)   
    assert all(np.abs((battery_energy_transition - battery_energy_transition_true)/battery_energy_transition) < 1e-3)

      
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
    tw_results = mission.evaluate()  
    
    # plot results
    tw_plot_mission(tw_results)   
    
    # save, load and plot old results 
    #save_tiltwing_results(tw_results)
    tw_old_results = load_tiltwing_results()
    tw_plot_mission(tw_old_results)   
   
    # RPM check during hover
    RPM        = tw_results.segments.hover.conditions.propulsion.rpm[0][0]
    RPM_true   = 574.0117525501447
    print(RPM) 
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  

    # lift Coefficient Check During Cruise
    lift_coefficient        = tw_results.segments.cruise.conditions.aerodynamics.lift_coefficient[0][0] 
    lift_coefficient_true   = 0.6509570689168221
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
    RPM_true   = 2258.2226911841635
    print(RPM) 
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  
    
    # Battery Energy Check During Transition
    battery_energy_hover_to_transition         = sr_results.segments.transition_1.conditions.propulsion.battery_energy[:,0]
    battery_energy_hover_to_transition_true    = np.array([3.06429126e+08, 3.06295041e+08, 3.05770150e+08, 3.04881939e+08,
                                                      3.03783051e+08, 3.02604903e+08, 3.01521913e+08, 3.00678374e+08,
                                                      3.00158036e+08, 2.99984153e+08])
    print(battery_energy_hover_to_transition)
    diff_battery_energy_hover_to_transition    = np.abs(battery_energy_hover_to_transition  - battery_energy_hover_to_transition_true) 
    print('battery_energy_hover_to_transition difference')
    print(diff_battery_energy_hover_to_transition)   
    assert all(np.abs((battery_energy_hover_to_transition - battery_energy_hover_to_transition_true)/battery_energy_hover_to_transition) < 1e-3)

    # lift Coefficient Check During Cruise
    lift_coefficient        = sr_results.segments.cruise.conditions.aerodynamics.lift_coefficient[0][0]
    lift_coefficient_true   = 0.696230824993908
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

def load_electric_helicopter_results():
    return SUAVE.Input_Output.SUAVE.load('results_electric_helicopter.res')

def save_electric_helicopter_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'results_electric_helicopter.res')
    return

if __name__ == '__main__': 
    main()    
