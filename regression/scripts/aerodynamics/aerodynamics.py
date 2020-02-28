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

import jax
import jax.numpy as np

#import numpy as np
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
                  
    # initalize the standard vlm aero model  
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.process.compute.lift.inviscid_wings.settings.use_surrogate             = True
    aerodynamics.process.compute.lift.inviscid_wings.settings.plot_vortex_distribution  = True
    aerodynamics.process.compute.lift.inviscid_wings.settings.plot_vehicle              = True 
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
    lift = state.conditions.aerodynamics.lift_coefficient
    lift_r = np.array([-0.81271571,-0.54878555,-0.50563137,-0.41216713,-0.2604832 ,
                       0.05574102, 0.29318675, 0.52096983, 0.81432677, 1.19037128, 1.20181071  ])[:,None]  
  
    print('lift = ', lift)
    
    lift_test = np.abs((lift-lift_r)/lift)
    
    print('\nCompute Lift Test Results\n')
    print(lift_test)
        
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
    cd_p_fuse      = drag_breakdown.parasite['fuselage'].parasite_drag_coefficient
    cd_p_wing      = drag_breakdown.parasite['main_wing'].parasite_drag_coefficient
    cd_tot         = drag_breakdown.total
   
    print('cd_m =', cd_m)
    
   
    (cd_c_r, cd_i_r, cd_m_r, cd_p_fuse_r, cd_p_wing_r, cd_tot_r) = reg_values()
    
    drag_tests = Data()
    drag_tests.cd_c = np.abs((cd_c-cd_c_r)/cd_c)
    for ii,cd in enumerate(drag_tests.cd_c):
        if np.isnan(cd):
            drag_tests.cd_c[ii] = np.abs((cd_c[ii]-cd_c_r[ii])/np.min(cd_c[cd_c!=0]))
    drag_tests.cd_i = np.abs((cd_i-cd_i_r)/cd_i)
    drag_tests.cd_m = np.abs((cd_m-cd_m_r)/cd_m)
    drag_tests.cd_p_fuse      = np.abs((cd_p_fuse - cd_p_fuse_r)/cd_p_fuse)
    drag_tests.cd_p_wing      = np.abs((cd_p_wing - cd_p_wing_r)/cd_p_wing)
    drag_tests.cd_tot         = np.abs((cd_tot - cd_tot_r)/cd_tot)
    
    print('\nCompute Drag Test Results\n')
    print('cd_tot=', cd_tot)
   
    for i, tests in list(drag_tests.items()): 
       
        assert(np.max(tests)<1e-4),'Aero regression test failed at ' + i
        
    #return conditions, configuration, geometry, test_num
      
def reg_values(): 
    cd_c_r = np.array([[8.10627624e-04,5.32238016e-08,2.86029027e-22,4.03345325e-09,2.65558332e-04,
                        1.72244621e-05,4.80688971e-10,6.20692196e-12,9.21190496e-06,8.35035892e-04,1.52647328e-14]]).T     
 
    cd_i_r = np.array([[0.02676262,0.01237179,0.01144083,0.00694995,0.00280848,
                        0.00012447,0.00354149,0.01163453,0.0272684 ,0.05744919,0.05960104 ]]).T 
            
    cd_m_r = np.array([[0.00117511,0.00117511,0.00117511,0.00117511,0.00117511,
                        0.00117511,0.00117511,0.00117511,0.00117511,0.00117511,0.00117511  ]]).T
   
    cd_p_fuse_r     = np.array([[0.00573497,0.00670141,0.01035105,0.00656857,0.00670352,
                                 0.00560765,0.00687999,0.00852837,0.00669708,0.00600831,0.00697568  ]]).T        
      
    cd_p_wing_r     = np.array([[0.00579887,0.00592795,0.00942986,0.0057326 ,0.00653004,
                                 0.00501665,0.00599058,0.00759737,0.00600963,0.00556283,0.00602578 ]]).T 
           
    cd_tot_r        = np.array([[0.04616812,0.03082864,0.03972979,0.02484412,0.02249526,
                                 0.01562698,0.02214446,0.03487413,0.04614441,0.07629951,0.07949967]]).T

    return cd_c_r, cd_i_r, cd_m_r, cd_p_fuse_r, cd_p_wing_r, cd_tot_r

if __name__ == '__main__':

    jit_main = jax.jit(main)
    jit_main()
    
    print('Aero regression test passed!')
      
