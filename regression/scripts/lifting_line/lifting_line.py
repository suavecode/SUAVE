# lifting_line.py
# 
# Created:  Nov 2017, E. Botero
# Modified: 
#

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
from SUAVE.Core import Data

import numpy as np

import copy, time
import random
from SUAVE.Attributes.Gases.Air import Air
import sys
#import vehicle file
sys.path.append('../Vehicles')
from Boeing_737 import vehicle_setup

# ----------------------------------------------------------------------
#   main
# ----------------------------------------------------------------------
def main():
    
    # initialize the vehicle
    vehicle = vehicle_setup() 
    for wing in vehicle.wings:
        wing.areas.wetted   = 2.0 * wing.areas.reference
        wing.areas.exposed  = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted  
        
    # initalize the aero model
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.process.compute.lift.inviscid_wings = SUAVE.Analyses.Aerodynamics.Lifting_Line()
    aerodynamics.geometry = vehicle
    aerodynamics.initialize()    
    
    #no of test points
    test_num = 11
    
    #specify the angle of attack
    angle_of_attacks = np.linspace(-.174,.174,test_num)[:,None] #* Units.deg
    
    # Cruise conditions (except Mach number)
    state = SUAVE.Analyses.Mission.Segments.Conditions.State()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    
    state.expand_rows(test_num)    
        
    # --------------------------------------------------------------------
    # Initialize variables needed for CL and CD calculations
    # Use a seeded random order for values
    # --------------------------------------------------------------------
    
    random.seed(1)
    Mc  = np.linspace(0.05,0.9,test_num)
    rho = np.linspace(0.3,1.3,test_num)
    mu  = np.linspace(5*10**-6,20*10**-6,test_num)
    T   = np.linspace(200,300,test_num)
    pressure = np.linspace(10**5,10**6,test_num)
    
    random.shuffle(Mc)
    random.shuffle(rho)
    random.shuffle(mu)
    random.shuffle(T)
    
    # Changed after to preserve seed for initial testing
    Mc  = Mc[:,None]
    rho = rho[:,None]
    mu  = mu[:,None]
    T   = T[:,None]
    pressure = pressure[:,None]
    
    air = Air()
    a   = air.compute_speed_of_sound(T,pressure)
    re  = rho*a*Mc/mu

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
    CL = results.lift.total
    CD = results.drag.total
    
    # --------------------------------------------------------------------
    # Test compute Lift
    # --------------------------------------------------------------------
    
    #compute_aircraft_lift(conditions, configuration, geometry) 
    
    lift = state.conditions.aerodynamics.lift_coefficient
    lift_r = np.array([-2.84689226, -1.06501674, -0.63426096, -0.35809118, -0.04487569,
        0.36343181, 0.61055156, 0.90742419, 1.43504496,  2.18401103,  1.81298486])[:,None]
    
    print 'lift = ', lift
    
    lift_test = np.abs((lift-lift_r)/lift)
    
    print '\nCompute Lift Test Results\n'
    #print lift_test
        
    assert(np.max(lift_test)<1e-4), 'Aero regression failed at compute lift test'    

if __name__ == '__main__':

    main()
    
    print 'Lifting Line test passed!'
      
