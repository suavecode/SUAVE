# mission_B737.py
# 
# Created:  Aug 2014, SUAVE Team
# Modified: Jun 2016, T. MacDonald

""" setup file for a mission with a 737
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

    # print weight breakdown
    #print_weight_breakdown(configs.base,filename = 'weight_breakdown.dat')

    # print engine data into file
    #print_engine_data(configs.base,filename = 'B737_engine_data.dat')

    # print parasite drag data into file
    # define reference condition for parasite drag
    ref_condition = Data()
    ref_condition.mach_number = 0.3
    ref_condition.reynolds_number = 12e6     
    #print_parasite_drag(ref_condition,configs.cruise,analyses,'B737_parasite_drag.dat')

    # print compressibility drag data into file
    #print_compress_drag(configs.cruise,analyses,filename = 'B737_compress_drag.dat')

    # print mission breakdown
    #print_mission_breakdown(results,filename='B737_mission_breakdown.dat')

    #load older results
    #save_results(results)
    #old_results = load_results()   

    # plt the old results
    # plot_mission(results)
    # plot_mission(old_results,'k-')
    # plt.show(block=True)
    # check the results
    #check_results(results,old_results)
    
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
 