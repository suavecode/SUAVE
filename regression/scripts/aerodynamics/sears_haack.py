# sears_haack.py
# 
# Created:  Feb 2021, T. MacDonald
# Modified: 

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------


import SUAVE
from SUAVE.Core import Units
from SUAVE.Core import Data

import numpy as np
import pylab as plt

import copy, time
import random
from SUAVE.Attributes.Gases.Air import Air
import sys
#import vehicle file
sys.path.append('../Vehicles')
from Concorde import vehicle_setup, configs_setup


def main():
    
    # initialize the vehicle
    vehicle = vehicle_setup() 
        
        
    # initalize the aero model
    aerodynamics = SUAVE.Analyses.Aerodynamics.Supersonic_Zero()      
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    aerodynamics.settings.span_efficiency = 0.95
    aerodynamics.settings.wave_drag_type = 'Sears-Haack'
    aerodynamics.settings.volume_wave_drag_scaling = 2.3 # calibrated to Concorde results  
        
    aerodynamics.initialize()    
    
    #no of test points
    test_num = 3
    
    #specify the angle of attack
    angle_of_attacks = np.linspace(-.0174,.0174*3,test_num)[:,None] #* Units.deg
    
    
    # Cruise conditions (except Mach number)
    state = SUAVE.Analyses.Mission.Segments.Conditions.State()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    
    
    state.expand_rows(test_num)    
        
    # --------------------------------------------------------------------
    # Initialize variables needed for CL and CD calculations
    # Use a pre-run random order for values
    # --------------------------------------------------------------------

    Mc = np.array([[1.03  ],
       [1.5 ],
       [2.0]])
    
    rho = np.array([[0.16],
           [0.16],
           [0.16]])
    
    mu = np.array([[1.42e-05],
           [1.42e-05],
           [1.42e-05]])
    
    T = np.array([[217.],
           [217.],
           [217.]])
    
    pressure = np.array([[ 10000.],
           [ 10000.],
           [ 10000.]])
    
    re = np.array([[6.0e6],
           [6.0e6],
           [6.0e6]])    
    
    air = Air()
    a = air.compute_speed_of_sound(T,pressure)
    
    re = rho*a*Mc/mu

    
    state.conditions.freestream.mach_number = Mc
    state.conditions.freestream.density = rho
    state.conditions.freestream.dynamic_viscosity = mu
    state.conditions.freestream.temperature = T
    state.conditions.freestream.pressure = pressure
    state.conditions.freestream.reynolds_number = re
    
    state.conditions.aerodynamics.angle_of_attack = angle_of_attacks   
    
    
    # --------------------------------------------------------------------
    # Surrogate
    # --------------------------------------------------------------------    
    
            
    #call the aero model        
    results = aerodynamics.evaluate(state)
    
    #build a polar for the markup aero
    polar = Data()    
    CL = results.lift.total
    CD = results.drag.total
    polar.lift = CL
    polar.drag = CD        
    
    # load older results
    #save_results(polar)
    old_polar = load_results()       
    
    # check the results
    check_results(polar,old_polar)
    
    return

def load_results():
    return SUAVE.Input_Output.SUAVE.load('sears_haack_results.res')

def save_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'sears_haack_results.res')
    return    

def check_results(new_results,old_results):

    # check segment values
    check_list = [
        'lift',
        'drag',
    ]

    # do the check
    for k in check_list:
        print(k)

        old_val = np.max( old_results.deep_get(k) )
        new_val = np.max( new_results.deep_get(k) )
        err = (new_val-old_val)/old_val
        print('Error at Max:' , err)
        assert np.abs(err) < 1e-6 , 'Max Check Failed : %s' % k

        old_val = np.min( old_results.deep_get(k) )
        new_val = np.min( new_results.deep_get(k) )
        err = (new_val-old_val)/old_val
        print('Error at Min:' , err)
        assert np.abs(err) < 1e-6 , 'Min Check Failed : %s' % k        

        print('')


    return
    
if __name__ == '__main__':

    main()