# test_aerodynamics
#
# Created:  Tim MacDonald - 09/09/14
# Modified: Tim MacDonald - 03/10/15
#
# Updated for new structure. cd_tot change assumed due to propulsion updates


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units
from SUAVE.Core import Data

from SUAVE.Methods.Aerodynamics.Supersonic_Zero.Lift import compute_aircraft_lift
from SUAVE.Methods.Aerodynamics.Supersonic_Zero.Drag import compute_aircraft_drag

import numpy as np
import pylab as plt

import copy, time
from copy import deepcopy
import random
import sys
#import vehicle file
sys.path.append('../Vehicles')
from Boeing_737 import vehicle_setup


def main():
    
    vehicle = vehicle_setup() # Create the vehicle for testing
    for wing in vehicle.wings:
        wing.areas.wetted   = 2.0 * wing.areas.reference
        wing.areas.exposed  = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted    
    aerodynamics = SUAVE.Analyses.Aerodynamics.Supersonic_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    vehicle.aerodynamics_model = aerodynamics   
    vehicle.aerodynamics_model.initialize()       
    
    test_num = 11 # Length of arrays used in this test
    
    # --------------------------------------------------------------------
    # Test Lift Surrogate
    # --------------------------------------------------------------------    
    
    AoA = np.linspace(-.174,.174,test_num)[:,None] # +- 10 degrees
    
    # Cruise conditions (except Mach number)
    state = SUAVE.Analyses.Mission.Segments.Conditions.State()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    
    state.expand_rows(test_num)     
    
    # --------------------------------------------------------------------
    # Initialize variables needed for CL and CD calculations
    # Use a seeded random order for values
    # --------------------------------------------------------------------
    
    random.seed(1)
    Mc = np.linspace(0.05,0.9,test_num)
    random.shuffle(Mc)
    AoA = AoA.reshape(test_num,1)
    Mc = Mc.reshape(test_num,1)
    
    rho = np.linspace(0.3,1.3,test_num)
    random.shuffle(rho)
    rho = rho.reshape(test_num,1)
    
    mu = np.linspace(5*10**-6,20*10**-6,test_num)
    random.shuffle(mu)
    mu = mu.reshape(test_num,1)
    
    T = np.linspace(200,300,test_num)
    random.shuffle(T)
    T = T.reshape(test_num,1)
    
    pressure = np.linspace(10**5,10**6,test_num)
    pressure = pressure.reshape(test_num,1)
    
    state.conditions.freestream.mach_number = Mc
    state.conditions.freestream.density = rho
    state.conditions.freestream.dynamic_viscosity = mu
    state.conditions.freestream.temperature = T
    state.conditions.freestream.pressure = pressure
    
    state.conditions.aerodynamics.angle_of_attack = AoA    
    
    
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
    
    
    # --------------------------------------------------------------------
    # Test compute Lift
    # --------------------------------------------------------------------
    
    #compute_aircraft_lift(conditions, configuration, geometry) 
    
    lift = state.conditions.aerodynamics.lift_coefficient
    
    # Truth value
    lift_r = np.array([-2.42489437, -0.90696416, -0.53991953, -0.3044834 ,  -0.03710598,
                       0.31061936 ,  0.52106899,  0.77407765,  1.22389024,  1.86240501,
                       1.54587835])[:,None]
    lift_r = lift_r.reshape(test_num,1)
    
    lift_test = np.abs((lift-lift_r)/lift)
    
    print('\nCompute Lift Test Results\n')
    print(lift_test)
        
    assert(np.max(lift_test)<1e-4), 'Supersonic Aero regression failed at compute lift test'    
    
    
    # --------------------------------------------------------------------
    # Test compute drag 
    # --------------------------------------------------------------------
    
    #compute_aircraft_drag(conditions, configuration, geometry)
    
    # Pull calculated values
    drag_breakdown = state.conditions.aerodynamics.drag_breakdown
    
    # Only one wing is evaluated since they rely on the same function
    cd_c           = drag_breakdown.compressible['main_wing'].compressibility_drag
    cd_i           = drag_breakdown.induced.total
    cd_m           = drag_breakdown.miscellaneous.total
    cd_m_fuse_base = drag_breakdown.miscellaneous.fuselage_base
    cd_m_fuse_up   = drag_breakdown.miscellaneous.fuselage_upsweep
    cd_m_nac_base  = drag_breakdown.miscellaneous.nacelle_base['turbo_fan']
    cd_m_ctrl      = drag_breakdown.miscellaneous.control_gaps
    cd_p_fuse      = drag_breakdown.parasite['fuselage'].parasite_drag_coefficient
    cd_p_wing      = drag_breakdown.parasite['main_wing'].parasite_drag_coefficient
    cd_tot         = drag_breakdown.total
    
    print(cd_c)
    print(cd_i)
    print(cd_m)
    print(cd_m_fuse_base)
    print(cd_m_fuse_up)
    print(cd_m_nac_base)
    print(cd_m_ctrl) 
    print(cd_p_fuse)
    print(cd_p_wing)
    print(cd_tot) 
    
    
    # Truth values
    (cd_c_r, cd_i_r, cd_m_r, cd_m_fuse_base_r, cd_m_fuse_up_r, cd_m_nac_base_r, cd_m_ctrl_r, cd_p_fuse_r, cd_p_wing_r, cd_tot_r) = reg_values()
    
    cd_c_r = cd_c_r.reshape(test_num,1)
    cd_i_r = cd_i_r.reshape(test_num,1)
    cd_m_r = cd_m_r.reshape(test_num,1)
    cd_m_fuse_base_r = cd_m_fuse_base_r.reshape(test_num,1)
    cd_m_fuse_up_r = cd_m_fuse_up_r.reshape(test_num,1)
    cd_m_nac_base_r = cd_m_nac_base_r.reshape(test_num,1)
    cd_m_ctrl_r = cd_m_ctrl_r.reshape(test_num,1)
    cd_p_fuse_r = cd_p_fuse_r.reshape(test_num,1)
    cd_p_wing_r = cd_p_wing_r.reshape(test_num,1)
    cd_tot_r = cd_tot_r.reshape(test_num,1)
    
    drag_tests = Data()
    drag_tests.cd_c = np.abs((cd_c-cd_c_r)/cd_c)
    drag_tests.cd_i = np.abs((cd_i-cd_i_r)/cd_i)
    drag_tests.cd_m = np.abs((cd_m-cd_m_r)/cd_m)
    # Line below is not normalized since regression values are 0, insert commented line if this changes
    drag_tests.cd_m_fuse_base = np.abs((cd_m_fuse_base-cd_m_fuse_base_r)) # np.abs((cd_m_fuse_base-cd_m_fuse_base_r)/cd_m_fuse_base)
    drag_tests.cd_m_fuse_up   = np.abs((cd_m_fuse_up - cd_m_fuse_up_r)/cd_m_fuse_up)
    drag_tests.cd_m_ctrl      = np.abs((cd_m_ctrl - cd_m_ctrl_r)/cd_m_ctrl)
    drag_tests.cd_p_fuse      = np.abs((cd_p_fuse - cd_p_fuse_r)/cd_p_fuse)
    drag_tests.cd_p_wing      = np.abs((cd_p_wing - cd_p_wing_r)/cd_p_wing)
    drag_tests.cd_tot         = np.abs((cd_tot - cd_tot_r)/cd_tot)
    
    print('\nCompute Drag Test Results\n')
    #print drag_tests
    
    for i, tests in list(drag_tests.items()):
        assert(np.max(tests)<1e-4),'Supersonic Aero regression test failed at ' + i
    
    #return state.conditions, configuration, geometry, test_num  


    
    
    
    #lift_model = vehicle.aerodynamics_model.surrogates.lift_coefficient_sub
    
    #wing_lift = lift_model(AoA)
    
    ## Truth value
    #wing_lift_r = np.array([-0.79420805, -0.56732369, -0.34043933, -0.11355497,  0.11332939,
                            #0.34021374,  0.5670981 ,  0.79398246,  1.02086682,  1.24775117,
                            #1.47463553])[:,None]
    
    #surg_test = np.abs((wing_lift-wing_lift_r)/wing_lift)
    
    #print 'Surrogate Test Results \n'
    #print surg_test
    
    #assert(np.max(surg_test)<1e-4), 'Supersonic Aero regression failed at surrogate test'

    
    ## --------------------------------------------------------------------
    ## Initialize variables needed for CL and CD calculations
    ## Use a seeded random order for values
    ## --------------------------------------------------------------------
    
    #random.seed(1)
    #Mc = np.linspace(0.05,0.9,test_num)
    #random.shuffle(Mc)
    #AoA = AoA.reshape(test_num,1)
    #Mc = Mc.reshape(test_num,1)
    
    #rho = np.linspace(0.3,1.3,test_num)
    #random.shuffle(rho)
    #rho = rho.reshape(test_num,1)
    
    #mu = np.linspace(5*10**-6,20*10**-6,test_num)
    #random.shuffle(mu)
    #mu = mu.reshape(test_num,1)
    
    #T = np.linspace(200,300,test_num)
    #random.shuffle(T)
    #T = T.reshape(test_num,1)
    
    #pressure = np.linspace(10**5,10**6,test_num)
    #pressure = pressure.reshape(test_num,1)
    
    #conditions = Data()
    
    #conditions.freestream = Data()
    #conditions.freestream.mach_number = Mc
    #conditions.freestream.density = rho
    #conditions.freestream.dynamic_viscosity = mu
    #conditions.freestream.temperature = T
    #conditions.freestream.pressure = pressure
    
    #conditions.aerodynamics = Data()
    #conditions.aerodynamics.angle_of_attack = AoA
    #conditions.aerodynamics.lift_breakdown = Data()
    #conditions.aerodynamics.lift_breakdown.inviscid_wings_lift = wing_lift
    
    #configuration = vehicle.aerodynamics_model.settings
    
    #conditions.aerodynamics.drag_breakdown = Data()

    #geometry = Data()
    #for k in ['fuselages','wings','propulsors']:
        #geometry[k] = deepcopy(vehicle[k])    
    #geometry.reference_area = vehicle.reference_area  
    ##geometry.wings[0] = Data()
    ##geometry.wings[0].vortex_lift = False
    
    ## --------------------------------------------------------------------
    ## Test compute Lift
    ## --------------------------------------------------------------------
    
    #compute_aircraft_lift(conditions, configuration, geometry) 
    
    #lift = conditions.aerodynamics.lift_breakdown.total
    
    ## Truth value
    #lift_r = np.array([-2.07712357, -0.73495391, -0.38858687, -0.1405849 ,  0.22295808,
                       #0.5075275 ,  0.67883681,  0.92787301,  1.40470556,  2.08126751,
                       #1.69661601])
    #lift_r = lift_r.reshape(test_num,1)
    
    #lift_test = np.abs((lift-lift_r)/lift)
    
    #print '\nCompute Lift Test Results\n'
    #print lift_test
        
    #assert(np.max(lift_test)<1e-4), 'Supersonic Aero regression failed at compute lift test'    
    
    
    ## --------------------------------------------------------------------
    ## Test compute drag 
    ## --------------------------------------------------------------------
    
    #compute_aircraft_drag(conditions, configuration, geometry)
    
    ## Pull calculated values
    #drag_breakdown = conditions.aerodynamics.drag_breakdown
    
    ## Only one wing is evaluated since they rely on the same function
    #cd_c           = drag_breakdown.compressible['main_wing'].compressibility_drag
    #cd_i           = drag_breakdown.induced.total
    #cd_m           = drag_breakdown.miscellaneous.total
    #cd_m_fuse_base = drag_breakdown.miscellaneous.fuselage_base
    #cd_m_fuse_up   = drag_breakdown.miscellaneous.fuselage_upsweep
    #cd_m_nac_base  = drag_breakdown.miscellaneous.nacelle_base['turbo_fan']
    #cd_m_ctrl      = drag_breakdown.miscellaneous.control_gaps
    #cd_p_fuse      = drag_breakdown.parasite['fuselage'].parasite_drag_coefficient
    #cd_p_wing      = drag_breakdown.parasite['main_wing'].parasite_drag_coefficient
    #cd_tot         = drag_breakdown.total
    
    ## Truth values
    #(cd_c_r, cd_i_r, cd_m_r, cd_m_fuse_base_r, cd_m_fuse_up_r, cd_m_nac_base_r, cd_m_ctrl_r, cd_p_fuse_r, cd_p_wing_r, cd_tot_r) = reg_values()
    
    #cd_c_r = cd_c_r.reshape(test_num,1)
    #cd_i_r = cd_i_r.reshape(test_num,1)
    #cd_m_r = cd_m_r.reshape(test_num,1)
    #cd_m_fuse_base_r = cd_m_fuse_base_r.reshape(test_num,1)
    #cd_m_fuse_up_r = cd_m_fuse_up_r.reshape(test_num,1)
    #cd_m_nac_base_r = cd_m_nac_base_r.reshape(test_num,1)
    #cd_m_ctrl_r = cd_m_ctrl_r.reshape(test_num,1)
    #cd_p_fuse_r = cd_p_fuse_r.reshape(test_num,1)
    #cd_p_wing_r = cd_p_wing_r.reshape(test_num,1)
    #cd_tot_r = cd_tot_r.reshape(test_num,1)
    
    #drag_tests = Data()
    #drag_tests.cd_c = np.abs((cd_c-cd_c_r)/cd_c)
    #drag_tests.cd_i = np.abs((cd_i-cd_i_r)/cd_i)
    #drag_tests.cd_m = np.abs((cd_m-cd_m_r)/cd_m)
    ## Line below is not normalized since regression values are 0, insert commented line if this changes
    #drag_tests.cd_m_fuse_base = np.abs((cd_m_fuse_base-cd_m_fuse_base_r)) # np.abs((cd_m_fuse_base-cd_m_fuse_base_r)/cd_m_fuse_base)
    #drag_tests.cd_m_fuse_up   = np.abs((cd_m_fuse_up - cd_m_fuse_up_r)/cd_m_fuse_up)
    #drag_tests.cd_m_ctrl      = np.abs((cd_m_ctrl - cd_m_ctrl_r)/cd_m_ctrl)
    #drag_tests.cd_p_fuse      = np.abs((cd_p_fuse - cd_p_fuse_r)/cd_p_fuse)
    #drag_tests.cd_p_wing      = np.abs((cd_p_wing - cd_p_wing_r)/cd_p_wing)
    #drag_tests.cd_tot         = np.abs((cd_tot - cd_tot_r)/cd_tot)
    
    #print '\nCompute Drag Test Results\n'
    #print drag_tests
    
    #for i, tests in drag_tests.items():
        #assert(np.max(tests)<1e-4),'Supersonic Aero regression test failed at ' + i
    
    #return conditions, configuration, geometry, test_num  
    
    

def reg_values():
    
    # Truth values from drag
    
    cd_c_r = np.array([  1.41429794e-08,   2.96579619e-09,   1.03047740e-22,   4.50771390e-09,
                         1.27784183e-03,   1.31214322e-04,   3.98984222e-09,   6.19742191e-11,
                         8.21182714e-05,   1.20217216e-03,   5.63926215e-14])
    
    cd_i_r = np.array([ 0.17295472,  0.02165349,  0.00605319,  0.00079229,  0.00199276,
                        0.01032588,  0.01847305,  0.03451317,  0.07910034,  0.17364551,
                        0.11539178])
    cd_m_r = np.array([ 0.00047933,  0.00047933,  0.00047933,  0.00047933,  0.00047933,
                        0.00047933,  0.00047933,  0.00047933,  0.00047933,  0.00047933,
                        0.00047933])
    
    cd_m_fuse_base_r = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
    
    cd_m_fuse_up_r   = np.array([  4.80530506e-05,   4.80530506e-05,   4.80530506e-05,
                                   4.80530506e-05,   4.80530506e-05,   4.80530506e-05,
                                   4.80530506e-05,   4.80530506e-05,   4.80530506e-05,
                                   4.80530506e-05,   4.80530506e-05])
    
    cd_m_nac_base_r = np.array([ 0.00033128,  0.00033128,  0.00033128,  0.00033128,  0.00033128,
                                0.00033128,  0.00033128,  0.00033128,  0.00033128,  0.00033128,
                                0.00033128])
    
    cd_m_ctrl_r     = np.array([ 0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,
                                 0.0001,  0.0001,  0.0001,  0.0001])
    
    cd_p_fuse_r     = np.array([  0.00816064,  0.00941302,  0.01437385,  0.00922505,  0.00946979,
                                  0.00792089,  0.00964896,  0.01190067,  0.00941451,  0.00848787,
                                  0.00977573])
    
    cd_p_wing_r     = np.array([ 0.00398269,  0.00401536,  0.00619387,  0.00388993,  0.00442375,
                                 0.00343623,  0.00405385,  0.00506457,  0.00406928,  0.00379353,
                                 0.00407611])
    
    cd_tot_r        = np.array([ 0.19422051,  0.03978096,  0.0333111 ,  0.01809441,  0.02202983,
                                 0.02565818,  0.03690062,  0.05755836,  0.09853024,  0.19460394,
                                 0.13595443])
    
    return cd_c_r[:,None], cd_i_r[:,None], cd_m_r[:,None], cd_m_fuse_base_r[:,None], cd_m_fuse_up_r[:,None], \
           cd_m_nac_base_r[:,None], cd_m_ctrl_r[:,None], cd_p_fuse_r[:,None], cd_p_wing_r[:,None], cd_tot_r[:,None]

if __name__ == '__main__':
    #(conditions, configuration, geometry, test_num) = main()
    main()
    
    print('Supersonic Aero regression test passed!')
    
    # --------------------------------------------------------------------
    # Drag Polar
    # --------------------------------------------------------------------  
    
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
    aerodynamics = SUAVE.Analyses.Aerodynamics.Supersonic_Zero()
    aerodynamics.geometry = vehicle
    
    ## modify inviscid wings - linear model
    #inviscid_wings = SUAVE.Analyses.Aerodynamics.Linear_Lift()
    #inviscid_wings.settings.slope_correction_coefficient = 1.04
    #inviscid_wings.settings.zero_lift_coefficient = 2.*np.pi* 3.1 * Units.deg    
    #aerodynamics.process.compute.lift.inviscid_wings = inviscid_wings
    
    ## modify inviscid wings - avl model
    #inviscid_wings = SUAVE.Analyses.Aerodynamics.Surrogates.AVL()
    #inviscid_wings.geometry = vehicle
    #aerodynamics.process.compute.lift.inviscid_wings = inviscid_wings
    
    aerodynamics.initialize()    
    
    
    #no of test points
    test_num = 11
    
    #specify the angle of attack
    AoA = np.linspace(-.174,.174,test_num)[:,None] #* Units.deg
    
    
    # Cruise conditions (except Mach number)
    state = SUAVE.Analyses.Mission.Segments.Conditions.State()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    
    
    state.expand_rows(test_num)   
    
    random.seed(1)
    Mc = np.linspace(0.05,0.9,test_num)
    random.shuffle(Mc)
    AoA = AoA.reshape(test_num,1)
    Mc = Mc.reshape(test_num,1)
    
    rho = np.linspace(0.3,1.3,test_num)
    random.shuffle(rho)
    rho = rho.reshape(test_num,1)
    
    mu = np.linspace(5*10**-6,20*10**-6,test_num)
    random.shuffle(mu)
    mu = mu.reshape(test_num,1)
    
    T = np.linspace(200,300,test_num)
    random.shuffle(T)
    T = T.reshape(test_num,1)
    
    pressure = np.linspace(10**5,10**6,test_num)
    pressure = pressure.reshape(test_num,1)    
    
    #specify  the conditions at which to perform the aerodynamic analysis
    state.conditions.aerodynamics.angle_of_attack = AoA #angle_of_attacks
    state.conditions.freestream.mach_number = Mc #np.array([0.8]*test_num)
    state.conditions.freestream.density = rho #np.array([0.3804534]*test_num)
    state.conditions.freestream.dynamic_viscosity = mu #np.array([1.43408227e-05]*test_num)
    state.conditions.freestream.temperature = T #np.array([218.92391647]*test_num)
    state.conditions.freestream.pressure = pressure #np.array([23908.73408391]*test_num)
            
    #call the aero model        
    results = aerodynamics.evaluate(state)
    
    #build a polar for the markup aero
    polar = Data()    
    CL = results.lift.total
    CD = results.drag.total
    polar.lift = CL
    polar.drag = CD
    

    ##load old results
    #old_polar = SUAVE.Input_Output.load('polar_M8.pkl') #('polar_old2.pkl')
    #CL_old = old_polar.lift
    #CD_old = old_polar.drag

    
    #plot the results
    plt.figure("Drag Polar")
    axes = plt.gca()     
    axes.plot(CD,CL,'bo-') #,CD_old,CL_old,'*')
    axes.set_xlabel('$C_D$')
    axes.set_ylabel('$C_L$')
    
    
    plt.show(block=True) # here so as to not block the regression test
