# test_Stopped_Rotor.py
# 
# Created:  Feb 2020, M. Clarke
#
""" setup file for a mission with a  Stopped Rotor eVTOL 
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units , Data 
import numpy as np 
import time 
import sys
  
from mission_Stopped_Rotor  import full_setup, plot_mission 

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():        
        
    # ------------------------------------------------------------------------------------------------------------------
    # Stopped-Rotor   
    # ------------------------------------------------------------------------------------------------------------------
    # build the vehicle, configs, and analyses
    configs, analyses = full_setup() 
    analyses.finalize()     
    weights   = analyses.weights
    breakdown = weights.evaluate() 
    mission   = analyses.mission  
    
    # evaluate mission     
    results   = mission.evaluate()
        
    # plot results
    plot_mission(results,configs)
      
    # save, load and plot old results 
    save_stopped_rotor_results(results)
    old_results = load_stopped_rotor_results()
    plot_mission(old_results,configs) 
 
    
    # RPM of rotor check during hover
    RPM        = results.segments.climb_1.conditions.propulsion.rpm_lift[0][0]
    RPM_true   = 2258.286261769841
    print(RPM) 
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  
    
    # Battery Energy Check During Transition
    battery_energy_hover_to_transition         = results.segments.transition_1.conditions.propulsion.battery_energy[:,0]
    battery_energy_hover_to_transition_true    = np.array([3.06429161e+08 3.06380479e+08 3.06186506e+08 3.05843629e+08
                                                           3.05384280e+08 3.04820654e+08 3.04178123e+08 3.03484506e+08
                                                           3.02775394e+08 3.02087748e+08 3.01458035e+08 3.00916095e+08
                                                           3.00483060e+08 3.00170388e+08 2.99982462e+08 2.99919877e+08])
    print(battery_energy_hover_to_transition)
    diff_battery_energy_hover_to_transition    = np.abs(battery_energy_hover_to_transition  - battery_energy_hover_to_transition_true) 
    print('battery_energy_hover_to_transition difference')
    print(diff_battery_energy_hover_to_transition)   
    assert all(np.abs((battery_energy_hover_to_transition - battery_energy_hover_to_transition_true)/battery_energy_hover_to_transition) < 1e-3)

    # lift Coefficient Check During Cruise
    lift_coefficient        = results.segments.cruise.conditions.aerodynamics.lift_coefficient[0][0]
    lift_coefficient_true   = 0.6962308249944807
    print(lift_coefficient)
    diff_CL                 = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-3    
    
    return

 
def load_stopped_rotor_results():
    return SUAVE.Input_Output.SUAVE.load('results_stopped_rotor.res')

def save_stopped_rotor_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'results_stopped_rotor.res')
    return
 

if __name__ == '__main__': 
    main()    
