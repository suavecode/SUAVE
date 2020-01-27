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
    # Use a pre-run random order for values
    # --------------------------------------------------------------------

    Mc = np.array([[0.9  ],
       [0.475],
       [0.05 ],
       [0.39 ],
       [0.815],
       [0.645],
       [0.305],
       [0.22 ],
       [0.56 ],
       [0.73 ],
       [0.135]])
    
    rho = np.array([[0.8],
           [1. ],
           [0.5],
           [1.1],
           [0.4],
           [1.3],
           [0.6],
           [0.3],
           [0.9],
           [0.7],
           [1.2]])
    
    mu = np.array([[1.85e-05],
           [1.55e-05],
           [1.40e-05],
           [1.10e-05],
           [2.00e-05],
           [8.00e-06],
           [6.50e-06],
           [9.50e-06],
           [1.70e-05],
           [1.25e-05],
           [5.00e-06]])
    
    T = np.array([[270.],
           [250.],
           [280.],
           [260.],
           [240.],
           [200.],
           [290.],
           [230.],
           [210.],
           [300.],
           [220.]])
    
    pressure = np.array([[ 100000.],
           [ 190000.],
           [ 280000.],
           [ 370000.],
           [ 460000.],
           [ 550000.],
           [ 640000.],
           [ 730000.],
           [ 820000.],
           [ 910000.],
           [1000000.]])
    
    re = np.array([[12819987.97468646],
           [ 9713525.47464844],
           [  599012.59815633],
           [12606549.94372309],
           [ 5062187.10214493],
           [29714816.00808047],
           [ 9611290.40694227],
           [ 2112171.68320523],
           [ 8612638.72342302],
           [14194381.78364854],
           [ 9633881.90543247]])    
    
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
    lift_r = np.array( [-2.37193091,-0.92896235,-0.60179935,-0.41769481,-0.29025691,
                        0.06311908, 0.27790296, 0.49317134, 0.84190929, 1.33725435, 1.14081317 ])[:,None]
   
    print('lift = ', lift)
    
    lift_test = np.abs((lift-lift_r)/lift)
    
    print('\nCompute Lift Test Results\n')
    #print lift_test
        
    assert(np.max(lift_test)<1e-6), 'Aero regression failed at compute lift test'    
    
    
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
   
    print('cd_m =', cd_m)
    
   
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
    
    print('\nCompute Drag Test Results\n')    
    print('cd_tot=', cd_tot)
   
    for i, tests in list(drag_tests.items()): 
       
        assert(np.max(tests)<1e-4),'Aero regression test failed at ' + i
        
    #return conditions, configuration, geometry, test_num
      

def reg_values():
    cd_c_r = np.array([[2.77841166e-09,9.54913908e-10,3.09611169e-23,1.00920702e-09,
                        9.88085452e-05,1.98202770e-05,9.14384129e-10,1.80326366e-11,
                        4.49667249e-05,4.00421207e-03,6.67689658e-14]]).T

           
    cd_i_r = np.array([[ 2.28100659e-01,3.54750736e-02,1.62244114e-02,7.14240342e-03,
                         3.48972960e-03,1.59691414e-04,3.18411319e-03,1.04353114e-02,
                         2.91672176e-02,7.25472784e-02,5.37425788e-02]]).T


           
    cd_m_r = np.array([[ 0.0011752,0.0011752,0.0011752,0.0011752,0.0011752,
                         0.0011752,0.0011752,0.0011752,0.0011752,0.0011752,
                         0.0011752]]).T

           
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

    cd_p_fuse_r     = np.array([[ 0.00573221,0.00669671,0.01034335,0.00656387,0.00669973,
                                  0.00560398,0.00687499,0.0085221 ,0.00669252,0.0060046 ,
                                  0.00697051]]).T

    cd_p_wing_r     = np.array([[ 0.00579887,0.00592795,0.00942986,0.0057326 ,
                                  0.00653004,0.00501665,0.00599058,0.00759737,
                                  0.00600963,0.00556283,0.00602578]]).T

           
    cd_tot_r        = np.array([[ 0.24973292,0.0544701 ,0.04474086,0.02511415,
                                  0.0228575 ,0.0157194 ,0.02185773,0.03375335,
                                  0.04819338,0.09493473,0.07360264]]).T
         
    return cd_c_r, cd_i_r, cd_m_r, cd_m_fuse_base_r, cd_m_fuse_up_r, \
           cd_m_nac_base_r, cd_m_ctrl_r, cd_p_fuse_r, cd_p_wing_r, cd_tot_r

if __name__ == '__main__':

    main()
    
    print('Aero regression test passed!')
      
