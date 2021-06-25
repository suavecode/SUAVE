import numpy                as np 
import matplotlib.pyplot    as plt

import SUAVE
from SUAVE.Core                                                     import Data, Units
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift           import VLM as VLM
from SUAVE.Plots.Geometry_Plots.plot_vehicle_vlm_panelization       import plot_vehicle_vlm_panelization

from regression.scripts.Vehicles.Boeing_737  import vehicle_setup   as b737_setup

def main():
    
    # control surface cases
    deflections = np.array([-10, 10, 20]) *Units.degrees
    
    # get settings and conditions
    conditions = get_conditions()      
    settings = get_settings()
    
    # create results object
    results     = Data()
    results.CL  = np.array([])
    results.CDi = np.array([])
    results.CM  = np.array([])
    
    # run VLM
    for deflection in deflections:
        geometry    = get_deflected_b737(deflection)
        data        = VLM(conditions, settings, geometry)
        
        plot_title  = "{}, deflection = {} degrees".format(geometry.tag, round(deflection/Units.degrees))
        plot_vehicle_vlm_panelization(geometry, plot_control_points=False, save_filename=plot_title)
        CL, CDi, CM, __, __, __, __, __, __, __ = data
        
        results.CL  = np.append(results.CL , CL.flatten() )
        results.CDi = np.append(results.CDi, CDi.flatten())
        results.CM  = np.append(results.CM , CM.flatten() )
        
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


def get_deflected_b737(deflection):  
    pos_def    = (deflection > 0)
    sla_def    = deflection if pos_def else 0.
    fla_def    = deflection if pos_def else 0.
    ail_def    = deflection
    ele_def    = -deflection
    
    vehicle = b737_setup()
    vehicle.wings['main_wing'            ].control_surfaces['slat'    ].deflection = sla_def 
    vehicle.wings['main_wing'            ].control_surfaces['flap'    ].deflection = fla_def 
    vehicle.wings['main_wing'            ].control_surfaces['aileron' ].deflection = ail_def  
    vehicle.wings['horizontal_stabilizer'].control_surfaces['elevator'].deflection = ele_def  
 
    return vehicle

def get_conditions():
    machs      = np.array([0.4  ,0.4  ,0.4  ,])
    altitudes  = np.array([5000 ,5000 ,5000 ,])  *Units.ft
    aoas       = np.array([-6.  ,0.   ,6.   ,])  *Units.degrees   
    
    conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    atmosphere                              = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    speeds_of_sound                         = atmosphere.compute_values(altitudes).speed_of_sound
    v_infs                                  = machs * speeds_of_sound.flatten()
    conditions.freestream.velocity          = np.atleast_2d(v_infs).T
    conditions.freestream.mach_number       = np.atleast_2d(machs).T   
    conditions.aerodynamics.angle_of_attack = np.atleast_2d(aoas).T
    
    return conditions

def get_settings():
    settings = Data()
    settings.wing_spanwise_vortices          = 7
    settings.wing_chordwise_vortices         = 4
    settings.fuselage_spanwise_vortices      = 7
    settings.fuselage_chordwise_vortices     = 4
        
    settings.propeller_wake_model            = None
    settings.spanwise_cosine_spacing         = False
    settings.model_fuselage                  = True
    settings.initial_timestep_offset         = 0.0
    settings.wake_development_time           = 0.0 
    settings.number_of_wake_timesteps        = 0.0
    settings.leading_edge_suction_multiplier = 1. 
    settings.discretize_control_surfaces     = True
                
    #misc settings
    settings.show_prints = False
    
    return settings


def load_results():
    return SUAVE.Input_Output.SUAVE.load('control_surfaces_vlm_results.res')

def save_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'control_surfaces_vlm_results.res')
    return


if __name__ == '__main__':

    main()
    plt.show()
    print('control_surface_vlm regression test passed!')
