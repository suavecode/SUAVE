# aerodynamics.py
# 
# Created:  Sep 2014, T. MacDonald
# Modified: Nov 2016, T. MacDonald
#           Apr 2020, M. Clarke

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
    aerodynamics.settings.number_spanwise_vortices  = 5
    aerodynamics.settings.number_chordwise_vortices = 2
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
    
    lift   = state.conditions.aerodynamics.lift_coefficient
    
    print('lift = \n', lift)
    print('\nCompute Lift Test Results\n')
    #print lift_test
    
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
   
    print('cd_m =\n', cd_m)
    
    #drag_tests = Data()
    ## Commented lines represent values not set by current drag functions, but to be recreated in the future
    # Line below is not normalized since regression values are 0, insert commented line if this changes
    # drag_tests.cd_m_fuse_base = np.abs((cd_m_fuse_base-cd_m_fuse_base_r)) # np.abs((cd_m_fuse_base-cd_m_fuse_base_r)/cd_m_fuse_base)
    # drag_tests.cd_m_fuse_up   = np.abs((cd_m_fuse_up - cd_m_fuse_up_r)/cd_m_fuse_up)
    # drag_tests.cd_m_ctrl      = np.abs((cd_m_ctrl - cd_m_ctrl_r)/cd_m_ctrl)
    
    print('\nCompute Drag Test Results\n')    
    print('cd_tot=', cd_tot)
    
    # --------------------------------------------------------------------
    # Process All Results 
    # --------------------------------------------------------------------
    results            = make_results_object(lift, cd_c, cd_i, cd_m, cd_p_fuse, cd_p_wing, cd_tot)
    
    save_results(results, SAVE=False)
    results_regression = load_results()
    
    test_results(results, results_regression)
    
    #return conditions, configuration, geometry, test_num
    return
        
      
# ----------------------------------------------------------------------
#   Make/Test Results Utility Functions
# ----------------------------------------------------------------------
def make_results_object(lift, cd_c, cd_i, cd_m, cd_p_fuse, cd_p_wing, cd_tot):
    results = Data()
    results.lift      = lift
    results.cd_c      = cd_c
    results.cd_i      = cd_i     
    results.cd_m      = cd_m     
    results.cd_p_fuse = cd_p_fuse
    results.cd_p_wing = cd_p_wing
    results.cd_tot    = cd_tot 
    return results

def test_results(results, results_regression):
    max_errors = [1e-6, 1e-4]
    for key in results.keys():
        #get values
        vals    = results[key]
        vals_tr = results_regression[key]
        
        #get error
        if key in ['cd_c']:
            for ii,val in enumerate(vals):
                if np.isnan(val):
                    vals[ii] = np.abs((vals[ii]-vals_tr[ii])/np.min(vals[vals!=0]))        
        else:
            errors     = (vals-vals_tr)/vals_tr
        
        #check error
        max_error = max_errors[0] if key in ['lift'] else max_errors[1]
        assert np.max(np.abs(errors)) < max_error, 'Failed at {} test, case {}'.format(key, np.argmax(np.abs(errors))+1)
        
    return

# ----------------------------------------------------------------------
#   Save/Load Utility Functions
# ----------------------------------------------------------------------
def regression_results_filename():
    return 'aerodynamics_results.res'

def load_results():
    return SUAVE.Input_Output.SUAVE.load(regression_results_filename())

def save_results(results, SAVE=False):
    if SAVE==True:
        print('!####! SAVING NEW REGRESSION RESULTS !####!')
        SUAVE.Input_Output.SUAVE.archive(results,regression_results_filename())        
    return

# ----------------------------------------------------------------------
#   Call Main
# ----------------------------------------------------------------------
if __name__ == '__main__':

    main()
    
    print('Aero regression test passed!')
      
