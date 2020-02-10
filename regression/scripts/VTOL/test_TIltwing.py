# test_Tiltwing.py
# 
# Created:  Feb 2020, M. Clarke
#
""" setup file for a mission with Tiltwing eVTOL  
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units , Data 
import numpy as np  
import time  
import sys 

from mission_Tiltwing     import full_setup, plot_mission  

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():   
      
    # ------------------------------------------------------------------------------------------------------------
    # Tiltwing CRM  
    # ------------------------------------------------------------------------------------------------------------
    # build the vehicle, configs, and analyses
    configs, analyses = full_setup() 
    configs.finalize()
    analyses.finalize()
    weights   = analyses.configs.base.weights
    breakdown = weights.evaluate()    
    mission   = analyses.missions.base
    
    # evaluate mission    
    results = mission.evaluate()  
    
    # plot results
    plot_mission(results)   
    
    # save, load and plot old results 
    #save_tiltwing_results(results)
    old_results = load_tiltwing_results()
    plot_mission(old_results)   
   
    # RPM check during hover
    RPM        = results.segments.hover.conditions.propulsion.rpm[0][0]
    RPM_true   = 574.0117525501447
    print(RPM) 
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  

    # lift Coefficient Check During Cruise
    lift_coefficient        = results.segments.cruise.conditions.aerodynamics.lift_coefficient[0][0] 
    lift_coefficient_true   = 0.6509570689168221
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

if __name__ == '__main__': 
    main()    
