# aerodynamics.py
# 
# Created:  Sep 2014, T. MacDonald
# Modified: Nov 2016, T. MacDonald
#
# Modified to match compressibility drag updates

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
from Boeing_737 import vehicle_setup


def main():
    
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
    
    
    # --------------------------------------------------------------------
    # Test compute Lift
    # --------------------------------------------------------------------
    
    
    
    #compute_aircraft_lift(conditions, configuration, geometry) 
    
    lift = state.conditions.aerodynamics.lift_coefficient
    lift_r = np.array( [ -2.17753919,  -0.7768714 ,  -0.41862788,  -0.16569318,0.1949377 ,
                        0.49528782,  0.67624325,  0.93239723,   1.41834084,  2.1078681 ,
                        1.72191103,])[:,None]
    
    print 'lift = ', lift
    
    lift_test = np.abs((lift-lift_r)/lift)
    
    print '\nCompute Lift Test Results\n'
    #print lift_test
        
    assert(np.max(lift_test)<1e-4), 'Aero regression failed at compute lift test'    
    
    
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
    # cd_m_fuse_base = drag_breakdown.miscellaneous.fuselage_base
    # cd_m_fuse_up   = drag_breakdown.miscellaneous.fuselage_upsweep
    # cd_m_nac_base  = drag_breakdown.miscellaneous.nacelle_base['turbofan']
    # cd_m_ctrl      = drag_breakdown.miscellaneous.control_gaps
    cd_p_fuse      = drag_breakdown.parasite['fuselage'].parasite_drag_coefficient
    cd_p_wing      = drag_breakdown.parasite['main_wing'].parasite_drag_coefficient
    cd_tot         = drag_breakdown.total
   
    print 'cd_m =', cd_m
    
   
    (cd_c_r, cd_i_r, cd_m_r, cd_m_fuse_base_r, cd_m_fuse_up_r, cd_m_nac_base_r, cd_m_ctrl_r, cd_p_fuse_r, cd_p_wing_r, cd_tot_r) = reg_values()
    
    drag_tests = Data()
    drag_tests.cd_c = np.abs((cd_c-cd_c_r)/cd_c)
    for ii,cd in enumerate(drag_tests.cd_c):
        if np.isnan(cd):
            drag_tests.cd_c[ii] = np.abs((cd_c[ii]-cd_c_r[ii])/np.min(cd_c[cd_c!=0]))
    drag_tests.cd_i = np.abs((cd_i-cd_i_r)/cd_i)
    drag_tests.cd_m = np.abs((cd_m-cd_m_r)/cd_m)
    ## Commented lines represent values not set by current drag functions, but to be recreated in the future
    # Line below is not normalized since regression values are 0, insert commented line if this changes
    # drag_tests.cd_m_fuse_base = np.abs((cd_m_fuse_base-cd_m_fuse_base_r)) # np.abs((cd_m_fuse_base-cd_m_fuse_base_r)/cd_m_fuse_base)
    # drag_tests.cd_m_fuse_up   = np.abs((cd_m_fuse_up - cd_m_fuse_up_r)/cd_m_fuse_up)
    # drag_tests.cd_m_ctrl      = np.abs((cd_m_ctrl - cd_m_ctrl_r)/cd_m_ctrl)
    drag_tests.cd_p_fuse      = np.abs((cd_p_fuse - cd_p_fuse_r)/cd_p_fuse)
    drag_tests.cd_p_wing      = np.abs((cd_p_wing - cd_p_wing_r)/cd_p_wing)
    drag_tests.cd_tot         = np.abs((cd_tot - cd_tot_r)/cd_tot)
    
    print '\nCompute Drag Test Results\n'    
    print 'cd_tot=', cd_tot
   
    for i, tests in drag_tests.items(): 
       
        assert(np.max(tests)<1e-4),'Aero regression test failed at ' + i
        
    #return conditions, configuration, geometry, test_num
      

def reg_values():
    cd_c_r = np.array([[  8.08247570e-09,  2.32439492e-09,  8.72099506e-23,  3.95356534e-09,
                          1.12715582e-03,  1.25533189e-04,  3.95906799e-09,  6.25239791e-11,
                          8.18124461e-05,  1.10331600e-03,   5.40850166e-14]]).T

    cd_i_r = np.array([[  0.19128965,  0.02465096,   0.00777093,   0.00111688,   
                         0.00156422,   0.00978113,   0.01873018,   0.03699081,   
                         0.0822499 ,   0.17923674,   0.12162074 ]]).T



    cd_m_r = np.array([[  0.0011513,     0.0011513,  0.0011513, 0.0011513,
                         0.0011513,  0.0011513, 0.0011513,0.0011513,
                         0.0011513, 0.0011513,0.0011513]]).T



    cd_m_fuse_base_r = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]).T

    cd_m_fuse_up_r   = np.array([[  4.80530506e-05,   4.80530506e-05,   4.80530506e-05,
                                   4.80530506e-05,   4.80530506e-05,   4.80530506e-05,
                                   4.80530506e-05,   4.80530506e-05,   4.80530506e-05,
                                   4.80530506e-05,   4.80530506e-05]]).T

    cd_m_nac_base_r = np.array([[ 0.00033128,  0.00033128,  0.00033128,  0.00033128,  0.00033128,
                                 0.00033128,  0.00033128,  0.00033128,  0.00033128,  0.00033128,
                                 0.00033128]]).T

    cd_m_ctrl_r     = np.array([[ 0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,
                                 0.0001,  0.0001,  0.0001,  0.0001]]).T

    cd_p_fuse_r     = np.array([[ 0.00573221 ,  0.00669671,  0.01034335,  0.00656387,  0.00669973,
                                 0.00560398,  0.00687499, 0.0085221,  0.00669252,  0.0060046,
                                 0.00697051]]).T

    cd_p_wing_r     = np.array([[ 0.00568318,  0.00574321,  0.00911242,  0.00555159,  0.00636405,
                                 0.0048708 ,  0.00579879,  0.00734795,  0.00582637,  0.0054087,
                                 0.00583051]]).T

    cd_tot_r        = np.array([[ 0.21162067, 0.04269817,   0.03486839,   0.01825581,   0.02122359, 
                                 0.0250517 ,   0.03696244,   0.05986016,    0.10164356,   0.20016137,  
                                 0.14207549,  ]]).T
    
    
    
    return cd_c_r, cd_i_r, cd_m_r, cd_m_fuse_base_r, cd_m_fuse_up_r, \
           cd_m_nac_base_r, cd_m_ctrl_r, cd_p_fuse_r, cd_p_wing_r, cd_tot_r

if __name__ == '__main__':

    main()
    
    print 'Aero regression test passed!'
      
