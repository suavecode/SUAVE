# test_AVL.py
# 
# Created:  May 2017, M. Clarke

""" setup file for a mission with a 737 using AVL
"""

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


sys.path.append('../B737')
# the analysis functions


from mission_B737 import full_setup, simple_sizing 

def main(): 
   
    configs, analyses = full_setup()
    
    modify_analyses(analyses,configs)
    simple_sizing(configs, analyses)

    configs.finalize()
    analyses.finalize()

  
 
    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()

    ref_condition = Data()
    ref_condition.mach_number = 0.3
    ref_condition.reynolds_number = 12e6     

    lift_coefficient = results.conditions.climb_1.aerodynamics.lift_coefficient[0]
    lift_coefficient_true = 0.64576527
    print lift_coefficient
    diff_CL = np.abs(lift_coefficient  - lift_coefficient_true) 
    print 'CL difference'
    print diff_CL
    assert np.abs(lift_coefficient  - lift_coefficient_true) < 1e-3
    
    moment_coefficient = results.conditions.climb_1.stability.static.CM[0][0]
    moment_coefficient_true = 0.018348545647711226
    print moment_coefficient
    diff_CM = np.abs(moment_coefficient - moment_coefficient_true)
    print 'CM difference'
    print diff_CM
    assert np.abs(moment_coefficient - moment_coefficient_true) < 1e-3    
 
    return

def modify_analyses(analyses,configs):
    
    aerodynamics = SUAVE.Analyses.Aerodynamics.AVL()
    stability = SUAVE.Analyses.Stability.AVL()
    aerodynamics.geometry = copy.deepcopy(configs.base)
    stability.geometry = copy.deepcopy(configs.base)
    
    aerodynamics.process.compute.lift.inviscid.training_file       = 'base_data_aerodynamics.txt'
    stability.training_file        = 'base_data_stability.txt'
    
    analyses.append(aerodynamics)
    analyses.append(stability)
     
    return 

if __name__ == '__main__': 
    main()    
 