# test_Multicopter.py
# 
# Created:  Feb 2020, M. Clarke
#
""" setup file for a mission with an Electic Multicopter
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units , Data
import numpy as np 
import time 
import sys 

from mission_Multicopter import full_setup, plot_mission 

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():  
    # ------------------------------------------------------------------------------------------------------------------
    # Electric Helicopter  
    # ------------------------------------------------------------------------------------------------------------------
    # build the vehicle, configs, and analyses 
    configs, analyses = full_setup() 
    analyses.finalize()    
    weights      = analyses.configs.base.weights
    breakdown    = weights.evaluate()     
    mission      = analyses.missions.base  
    results   = mission.evaluate()
        
    # plot results
    plot_mission(results)
    
    # save, load and plot old results 
    #save_multicopter_results(results)
    old_results = load_multicopter_results()
    plot_mission(old_results) 
 
    
    # RPM of rotor check during hover
    RPM        = results.segments.climb.conditions.propulsion.rpm[0][0]
    RPM_true   = 1598.7692155337331
    print(RPM) 
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  
    
    # Battery Energy Check During Transition
    battery_energy_transition         = results.segments.hover.conditions.propulsion.battery_energy[:,0]
    battery_energy_transition_true    = np.array([3.57097712e+08, 3.56795677e+08, 3.55902186e+08, 3.54454664e+08,
                                                  3.52514086e+08, 3.50162799e+08, 3.47501414e+08, 3.44644776e+08,
                                                  3.41717150e+08, 3.38846818e+08, 3.36160348e+08, 3.33776835e+08,
                                                  3.31802409e+08, 3.30325288e+08, 3.29411640e+08, 3.29102462e+08])
    print(battery_energy_transition)
    diff_battery_energy_transition    = np.abs(battery_energy_transition  - battery_energy_transition_true) 
    print('battery energy of transition')
    print(diff_battery_energy_transition)   
    assert all(np.abs((battery_energy_transition - battery_energy_transition_true)/battery_energy_transition) < 1e-3)

 
    return

def load_multicopter_results():
    return SUAVE.Input_Output.SUAVE.load('results_multicopter.res')

def save_multicopter_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'results_multicopter.res')
    return

if __name__ == '__main__': 
    main()    
