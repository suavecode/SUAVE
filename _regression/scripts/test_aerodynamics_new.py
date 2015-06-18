# test_aerodynamics
#
# Created:  Trent Lukaczyk                : March 2015
# Modified: Anil Variyar, Trent Lukaczyk  : April 2015
#


import SUAVE
from SUAVE.Core import Units
from SUAVE.Core import Data

from test_mission_B737 import vehicle_setup

import numpy as np
import pylab as plt

import copy, time
from copy import deepcopy
import random

def main():
    
# --------------------------------------------------------------------
# Drag Polar
# --------------------------------------------------------------------

    # initialize the vehicle
    vehicle = vehicle_setup() 
    for wing in vehicle.wings:
        wing.areas.wetted   = 2.0 * wing.areas.reference
        wing.areas.exposed  = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted  
        
        
    # initalize the aero model
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    
    ## modify inviscid wings - linear model
    #inviscid_wings = SUAVE.Analyses.Aerodynamics.Linear_Lift()
    #inviscid_wings.settings.slope_correction_coefficient = 1.04
    #inviscid_wings.settings.zero_lift_coefficient = 2.*np.pi* 3.1 * Units.deg    
    #aerodynamics.process.compute.lift.inviscid_wings = inviscid_wings
    
    # modify inviscid wings - avl model
    inviscid_wings = SUAVE.Analyses.Aerodynamics.Surrogates.AVL()
    inviscid_wings.geometry = vehicle
    aerodynamics.process.compute.lift.inviscid_wings = inviscid_wings
    
    aerodynamics.initialize()    
    
    
    #no of test points
    test_num = 11
    
    #specify the angle of attack
    angle_of_attacks = np.linspace(-.174,.174,test_num) #* Units.deg
    
    
    # Cruise conditions (except Mach number)
    state = SUAVE.Analyses.Mission.Segments.Conditions.State()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    
    
    state.expand_rows(test_num)    
    
    #specify  the conditions at which to perform the aerodynamic analysis
    state.conditions.aerodynamics.angle_of_attack[:,0] = angle_of_attacks
    state.conditions.freestream.mach_number = np.array([0.8]*test_num)
    state.conditions.freestream.density = np.array([0.3804534]*test_num)
    state.conditions.freestream.dynamic_viscosity = np.array([1.43408227e-05]*test_num)
    state.conditions.freestream.temperature = np.array([218.92391647]*test_num)
    state.conditions.freestream.pressure = np.array([23908.73408391]*test_num)
            
    #call the aero model        
    results = aerodynamics.evaluate(state)
    
    #build a polar for the markup aero
    polar = Data()    
    CL = results.lift.total
    CD = results.drag.total
    polar.lift = CL
    polar.drag = CD
    

    #load old results
    old_polar = SUAVE.Input_Output.load('polar_M8.pkl') #('polar_old2.pkl')
    CL_old = old_polar.lift
    CD_old = old_polar.drag

    
    #plot the results
    plt.figure("Drag Polar")
    axes = plt.gca()     
    axes.plot(CD,CL,'bo-',CD_old,CL_old,'*')
    axes.set_xlabel('$C_D$')
    axes.set_ylabel('$C_L$')
    
    
    plt.show(block=True) # here so as to not block the regression test
      

if __name__ == '__main__':
    main()
    
