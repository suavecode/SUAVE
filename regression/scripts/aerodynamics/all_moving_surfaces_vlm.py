# control_surfaces_vlm.py
# 
# Created:  July 2021, A. Blaufox
# Modified: 
# 
# File to test all-moving surfaces in VLM

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import sys
import numpy as np 

import SUAVE
from SUAVE.Core                                                     import Data, Units
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift           import VLM as VLM
from SUAVE.Plots.Geometry.plot_vehicle_vlm_panelization             import plot_vehicle_vlm_panelization

sys.path.append('../Vehicles')

from All_Moving_Test_Bench import vehicle_setup as test_bench_setup
import matplotlib.pyplot                as plt

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    # all-moving surface deflection cases
    deflection_configs = get_array_of_deflection_configs()
    
    # get settings and conditions
    conditions = get_conditions()      
    settings   = get_settings()
    n_cases    = len(conditions.freestream.mach_number)
    
    # create results objects    
    results        = Data()
    results.CL     = np.empty(shape=[0,n_cases])
    results.CDi    = np.empty(shape=[0,n_cases])
    results.CM     = np.empty(shape=[0,n_cases])
    results.CYTOT  = np.empty(shape=[0,n_cases])
    results.CRMTOT = np.empty(shape=[0,n_cases])
    results.CYMTOT = np.empty(shape=[0,n_cases])
    
    # run VLM
    for i,deflection_config in enumerate(deflection_configs):
        geometry    = test_bench_setup(deflection_config=deflection_config)
        data        = VLM(conditions, settings, geometry)
        
        plot_title  = "Deflection Configuration #{}".format(i+1)
        plot_vehicle_vlm_panelization(geometry, plot_control_points=False, save_filename=plot_title)        
        
        results.CL         = np.vstack((results.CL     , data.CL.flatten()    ))
        results.CDi        = np.vstack((results.CDi    , data.CDi.flatten()   ))
        results.CM         = np.vstack((results.CM     , data.CM.flatten()    ))
        results.CYTOT      = np.vstack((results.CYTOT  , data.CYTOT.flatten() ))
        results.CRMTOT     = np.vstack((results.CRMTOT , data.CRMTOT.flatten()))
        results.CYMTOT     = np.vstack((results.CYMTOT , data.CYMTOT.flatten()))      
        
    # save/load results
    #save_results(results)
    results_tr = load_results()
    
    # check results
    for key in results.keys():
        vals    = results[key]
        vals_tr = results_tr[key]
        errors  = (vals-vals_tr)/vals_tr
        
        print('results.{}:'.format(key))
        print(vals)
        print('results_tr.{}:'.format(key))
        print(vals_tr) 
        print('errors:')
        print(errors)
        print('           ')
        
        max_err = np.max(np.abs(errors))
        assert max_err < 1e-6 , 'Failed at {} test'.format(key)
    
    return

# ----------------------------------------------------------------------
#   Setup Functions
# ----------------------------------------------------------------------
def get_array_of_deflection_configs():  
    stabilator_sign_duplicates   = [ 1., -1.,  0.]
    v_tail_right_sign_duplicates = [-1.,  0.,  0.]
    
    stabilator_hinge_fractions   = [0.25, 0.5, 0.75]
    v_tail_right_hinge_fractions = [0.75, 0.0, 0.25]
    
    stabilator_use_constant_hinge_fractions   = [False, False, False]
    v_tail_right_use_constant_hinge_fractions = [False, True , False]
    
    zero_vec = np.array([0.,0.,0.])
    stabilator_hinge_vectors     = [zero_vec*1, zero_vec*1, zero_vec*1]
    v_tail_right_hinge_vectors   = [zero_vec*1, zero_vec*1, np.array([0.,1.,0.])]
    
    deflections                  = [ 0.,  0.,  0.]
    stab_defs                    = [10.,-10., 30.]
    vt_r_defs                    = [10.,-10.,-30.]
    
    n_configs = len(deflections)
    deflection_configs = [Data() for i in range(n_configs)]
    
    for i, deflection_config in enumerate(deflection_configs):
        deflection_config.  stabilator_sign_duplicate =   stabilator_sign_duplicates[i]  
        deflection_config.v_tail_right_sign_duplicate = v_tail_right_sign_duplicates[i]
        
        deflection_config.  stabilator_hinge_fraction =   stabilator_hinge_fractions[i] 
        deflection_config.v_tail_right_hinge_fraction = v_tail_right_hinge_fractions[i]
        
        deflection_config.  stabilator_use_constant_hinge_fraction =   stabilator_use_constant_hinge_fractions[i]
        deflection_config.v_tail_right_use_constant_hinge_fraction = v_tail_right_use_constant_hinge_fractions[i]
        
        deflection_config.  stabilator_hinge_vector   =   stabilator_hinge_vectors[i]
        deflection_config.v_tail_right_hinge_vector   = v_tail_right_hinge_vectors[i]      
        
        deflection_config.deflection                  = deflections[i]                 
        deflection_config.stab_def                    = stab_defs[i]                   
        deflection_config.vt_r_def                    = vt_r_defs[i] 

    return deflection_configs

def get_conditions():
    machs      = np.array([0.4  ,0.4  ,0.4  ,0.4  ,1.4  ,])
    altitudes  = np.array([5000 ,5000 ,5000 ,5000 ,5000 ,])  *Units.ft
    aoas       = np.array([0.   ,6.   ,6.   ,0.   ,6    ,])  *Units.degrees #angle of attack in degrees
    PSIs       = np.array([3.   ,5.   ,0.   ,0.   ,5.   ,])  *Units.degrees #sideslip angle  in degrees
    PITCHQs    = np.array([3.   ,6.   ,0.   ,0.   ,6.   ,])  *Units.degrees #pitch rate      in degrees/s   
    ROLLQs     = np.array([3.   ,6.   ,0.   ,0.   ,6.   ,])  *Units.degrees #roll  rate      in degrees/s
    YAWQs      = np.array([3.   ,6.   ,0.   ,0.   ,6.   ,])  *Units.degrees #yaw   rate      in degrees/s       
    
    conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    atmosphere                              = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    speeds_of_sound                         = atmosphere.compute_values(altitudes).speed_of_sound
    v_infs                                  = machs * speeds_of_sound.flatten()
    conditions.freestream.velocity          = np.atleast_2d(v_infs).T
    conditions.freestream.mach_number       = np.atleast_2d(machs).T   
    conditions.aerodynamics.angle_of_attack = np.atleast_2d(aoas).T
    conditions.aerodynamics.side_slip_angle = np.atleast_2d(PSIs).T
    conditions.stability.dynamic.pitch_rate = np.atleast_2d(PITCHQs).T
    conditions.stability.dynamic.roll_rate  = np.atleast_2d(ROLLQs).T
    conditions.stability.dynamic.yaw_rate   = np.atleast_2d(YAWQs).T
    
    return conditions

def get_settings():
    settings = SUAVE.Analyses.Aerodynamics.Vortex_Lattice().settings
    settings.number_spanwise_vortices        = None
    settings.number_chordwise_vortices       = None  
    settings.wing_spanwise_vortices          = 5
    settings.wing_chordwise_vortices         = 4
    settings.fuselage_spanwise_vortices      = 5
    settings.fuselage_chordwise_vortices     = 4
        
    settings.use_bemt_wake_model             = False
    settings.propeller_wake_model            = None
    settings.spanwise_cosine_spacing         = False
    settings.model_fuselage                  = True
    settings.model_nacelle                   = True
    settings.initial_timestep_offset         = 0.0
    settings.wake_development_time           = 0.0 
    settings.number_of_wake_timesteps        = 0.0
    settings.leading_edge_suction_multiplier = 1. 
    settings.discretize_control_surfaces     = True
    settings.use_VORLAX_matrix_calculation   = False    
                
    #misc settings
    settings.show_prints = False
    
    return settings

# ----------------------------------------------------------------------
#   Save/Load Utility Functions
# ----------------------------------------------------------------------
def load_results():
    return SUAVE.Input_Output.SUAVE.load('all_moving_surfaces_vlm_results.res')

def save_results(results):
    print('!####! SAVING NEW REGRESSION RESULTS !####!')
    SUAVE.Input_Output.SUAVE.archive(results,'all_moving_surfaces_vlm_results.res')
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    
if __name__ == '__main__':
    main()
    plt.show()
