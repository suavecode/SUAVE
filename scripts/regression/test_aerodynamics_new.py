# test_aerodynamics
#
# Created:  Tim MacDonald - 09/09/14
# Modified: Tim MacDonald - 03/10/15
#
# Changed to use new structures and drag functions from Tarik

import SUAVE
from SUAVE.Core import Units
from SUAVE.Core import Data

#from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_aircraft_lift
#from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag import compute_aircraft_drag

from test_mission_B737 import vehicle_setup

import numpy as np
import pylab as plt

import copy, time
from copy import deepcopy
import random

SUAVE.Analyses.Process.verbose = True

def main():
    
    vehicle = vehicle_setup() # Create the vehicle for testing
    for wing in vehicle.wings:
        wing.areas.wetted   = 2.0 * wing.areas.reference
        wing.areas.exposed  = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted    
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    #vehicle.aerodynamics_model = aerodynamics   
    #vehicle.aerodynamics_model.finalize()
    
    test_num = 11 # Length of arrays used in this test
    
    # --------------------------------------------------------------------
    # Test Lift Surrogate
    # --------------------------------------------------------------------    
    
    AoA = np.linspace(-.174,.174,test_num) # +- 10 degrees
    AoA = AoA[:,None]
    
    
    # --------------------------------------------------------------------
    # Initialize variables needed for CL and CD calculations
    # Use a seeded random order for values
    # --------------------------------------------------------------------
    
    random.seed(1)
    Mc = np.linspace(0.05,0.9,test_num)
    random.shuffle(Mc)
    rho = np.linspace(0.3,1.3,test_num)
    random.shuffle(rho)
    mu = np.linspace(5*10**-6,20*10**-6,test_num)
    random.shuffle(mu)
    T = np.linspace(200,300,test_num)
    random.shuffle(T)
    pressure = np.linspace(10**5,10**6,test_num)
    
    # Changed after to preserve seed for initial testing
    Mc = Mc[:,None]
    rho = rho[:,None]
    mu = mu[:,None]
    T = T[:,None]
    pressure = pressure[:,None]

    
    conditions = Data()
    
    conditions.freestream = Data()
    conditions.freestream.mach_number = Mc
    conditions.freestream.density = rho
    conditions.freestream.dynamic_viscosity = mu
    conditions.freestream.temperature = T
    conditions.freestream.pressure = pressure
    
    conditions.aerodynamics = Data()
    conditions.aerodynamics.angle_of_attack = AoA
    #conditions.aerodynamics.lift_breakdown = Data()
    #conditions.aerodynamics.lift_breakdown.inviscid_wings_lift = wing_lift
    
    #configuration = vehicle.aerodynamics_model.settings
     

    
    return conditions, configuration, geometry, test_num
      

if __name__ == '__main__':
    #(conditions, configuration, geometry, test_num) = main()
    #(conditions, configuration, geometry, test_num) = main()
    
    #print 'Aero regression test passed!'
    
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
    aerodynamics.initialize()    


    test_num = 11
    
    angle_of_attacks = np.linspace(-.174,.174,test_num) #* Units.deg
    # Cruise conditions (except Mach number)
    state = SUAVE.Analyses.Mission.Segments.Conditions.State()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    
    
    state.expand_rows(test_num)    
    
    
    state.conditions.aerodynamics.angle_of_attack[:,0] = angle_of_attacks
    state.conditions.freestream.mach_number = np.array([0.8]*test_num)
    state.conditions.freestream.density = np.array([0.3804534]*test_num)
    state.conditions.freestream.dynamic_viscosity = np.array([1.43408227e-05]*test_num)
    state.conditions.freestream.temperature = np.array([218.92391647]*test_num)
    state.conditions.freestream.pressure = np.array([23908.73408391]*test_num)
    
    #compute_aircraft_lift(conditions, configuration, geometry) # geometry is third variable, not used
    #CL = conditions.aerodynamics.lift_breakdown.total    
    
    #compute_aircraft_drag(conditions, configuration, geometry)
    #CD = conditions.aerodynamics.drag_breakdown.total
        
        
    #print state.conditions.aerodynamics.drag_breakdown.compressible
    results = aerodynamics.evaluate(state)
    
    polar = Data()    
    
    CL = results.lift.total
    
 
    
    CD = results.drag.total
    
    polar.lift = CL
    polar.drag = CD
    
    old_polar = SUAVE.Input_Output.load('polar_M8.pkl') #('polar_old2.pkl')
    CL_old = old_polar.lift
    CD_old = old_polar.drag
    
    #print state.conditions.aerodynamics.drag_breakdown.compressible.total
    #print results
    print "\n---------------------------------\n"
    #print old_polar.drag_breakdown.compressible  #.compressible.compressibility_drag
    
    plt.figure("Drag Polar")
    axes = plt.gca()     
    axes.plot(CD,CL,'bo-',CD_old,CL_old,'*')
    axes.set_xlabel('$C_D$')
    axes.set_ylabel('$C_L$')
    
    
    plt.show(block=True) # here so as to not block the regression test