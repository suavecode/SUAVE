# sideslip_and_rotation_vlm.py
# 
# Created:  July 2021, A. Blaufox
# Modified: 
# 
# File to test sideslip and rotation rates (pith, roll, yaw) in VLM

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

from Boeing_737  import vehicle_setup   as b737_setup
import matplotlib.pyplot                as plt

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():    
    # get settings and conditions
    conditions = get_conditions()      
    settings   = get_settings()
    
    # run VLM
    geometry    = b737_setup()
    data        = VLM(conditions, settings, geometry)
    
    plot_title  = geometry.tag
    plot_vehicle_vlm_panelization(geometry, plot_control_points=False, save_filename=plot_title)

    # save/load results
    results = Data()
    results.CL         =  data.CL    
    results.CDi        =  data.CDi   
    results.CM         =  data.CM  
    results.CYTOT      =  data.CYTOT
    results.CRTOT      =  data.CRTOT
    results.CRMTOT     =  data.CRMTOT
    results.CNTOT      =  data.CNTOT
    results.CYMTOT     =  data.CYMTOT
    
    #save_results(results)
    results_tr = load_results()
    
    for key in results.keys():
        vals    = results[key]
        vals_tr = results_tr[key]
        errors  = (vals-vals_tr)/vals_tr
        print('{} errors:'.format(key)    )
        print(errors                      )
        print('                          ')
                
        max_err    = np.max(   np.abs(errors))
        argmax_err = np.argmax(np.abs(errors))
        assert max_err < 1e-6 , print('Failed at {} test, case {}'.format(key, argmax_err+1))
    
    return

# ----------------------------------------------------------------------
#   Setup Functions
# ----------------------------------------------------------------------
def get_conditions():
    machs      = np.array([0.4  ,0.4  ,0.4  ,0.4  ,0.4  ,1.4  ,])
    altitudes  = np.array([5000 ,5000 ,5000 ,5000 ,5000 ,5000 ,])  *Units.ft
    aoas       = np.array([-6.  ,1.   ,6.   ,1.   ,1.   ,6    ,])  *Units.degrees #angle of attack in degrees
    PSIs       = np.array([-5.  ,3.   ,5.   ,5.   ,0.   ,5.   ,])  *Units.degrees #sideslip angle  in degrees
    PITCHQs    = np.array([-6.  ,3.   ,6.   ,0.   ,5.   ,6.   ,])  *Units.degrees #pitch rate      in degrees/s   
    ROLLQs     = np.array([-6.  ,3.   ,6.   ,0.   ,5.   ,6.   ,])  *Units.degrees #roll  rate      in degrees/s
    YAWQs      = np.array([-6.  ,3.   ,6.   ,0.   ,5.   ,6.   ,])  *Units.degrees #yaw   rate      in degrees/s       
    
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
    settings.number_spanwise_vortices        = 7 
    settings.number_chordwise_vortices       = 4   
    settings.use_bemt_wake_model             = False
    settings.propeller_wake_model            = None
    settings.spanwise_cosine_spacing         = False
    settings.model_fuselage                  = True
    settings.model_nacelle                   = False
    settings.initial_timestep_offset         = 0.0
    settings.wake_development_time           = 0.0 
    settings.number_of_wake_timesteps        = 0.0
    settings.leading_edge_suction_multiplier = 1. 
    settings.discretize_control_surfaces     = False
    settings.use_VORLAX_matrix_calculation   = False    
                
    #misc settings
    settings.show_prints = False
    
    return settings

# ----------------------------------------------------------------------
#   Save/Load Utility Functions
# ----------------------------------------------------------------------
def load_results():
    return SUAVE.Input_Output.SUAVE.load('sideslip_and_rotation_vlm_results.res')

def save_results(results):
    print('!####! SAVING NEW REGRESSION RESULTS !####!')
    SUAVE.Input_Output.SUAVE.archive(results,'sideslip_and_rotation_vlm_results.res')
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    
if __name__ == '__main__':
    main()
    plt.show()
    print('sideslip_and_rotation_vlm regression test passed!')