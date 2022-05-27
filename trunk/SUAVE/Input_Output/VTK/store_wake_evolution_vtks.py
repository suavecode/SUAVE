## @ingroup Input_Output-VTK
# store_wake_evolution_vtks.py
#
# Created:  Feb 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data
from SUAVE.Input_Output.VTK.save_vehicle_vtk import save_vehicle_vtks
from SUAVE.Input_Output.VTK.save_evaluation_points_vtk import save_evaluation_points_vtk

import numpy as np

## @ingroup Input_Output-VTK
def store_wake_evolution_vtks(vehicle,vlm_results=None,save_loc=None,generate_vtks=True,time_step=None, verbose=False):
    """
    Saves evolution of rotor wake over single rotation. Outputs VTK files in legacy format.

    Inputs:
       wake                SUAVE Fidelity One rotor wake
       rotor               SUAVE rotor         

    Outputs:
       N/A

    Properties Used:
       N/A

    Assumptions:
       Fidelity-One rotor wake

    Source:
       None

    """
    
    # Unpack rotor
    prop_keys = vehicle.networks.battery_propeller.propellers.keys()
    Na        = vehicle.networks.battery_propeller.propellers[list(prop_keys)[0]].outputs.number_azimuthal_stations
    try:
        VD_wing   = vehicle.vortex_distribution
        wingPresent = True
    except:
        wingPresent = False
    if time_step == None:
        time_steps=Na
        time_step = 0
    else:
        time_steps = 1
    
    # --------------------------------------------------------------------------------------------------------------
    #    Store VTKs after wake is generated
    # --------------------------------------------------------------------------------------------------------------      
    if generate_vtks:
        if save_loc == None:
            pass
        else:
            # after converged, store vtks for final wake shape for each of Na starting positions
            for i in range(time_steps):
                for rotor in vehicle.networks.battery_propeller.propellers:
                    rotor.start_angle = i*2*np.pi / time_steps   
                    wake = rotor.Wake
                    
                    # extract rotor outputs
                    rotor_outputs = rotor.outputs   
                    
                    # Store evaluation points on this prop
                    Yb   = wake.vortex_distribution.reshaped_wake.Yblades_cp[i,0,0,:,0] 
                    Zb   = wake.vortex_distribution.reshaped_wake.Zblades_cp[i,0,0,:,0] 
                    Xb   = wake.vortex_distribution.reshaped_wake.Xblades_cp[i,0,0,:,0] 
                    
                    VD_rot = Data()
                    VD_rot.YC = (Yb[1:] + Yb[:-1])/2
                    VD_rot.ZC = (Zb[1:] + Zb[:-1])/2
                    VD_rot.XC = (Xb[1:] + Xb[:-1])/2
        
                    points = Data()
                    points.XC = VD_rot.XC
                    points.YC = VD_rot.YC
                    points.ZC = VD_rot.ZC
                    points.induced_velocities = Data()
                    points.induced_velocities.va = rotor_outputs.disc_axial_induced_velocity[0,:,i]
                    points.induced_velocities.vt = rotor_outputs.disc_tangential_induced_velocity[0,:,i]
                    save_evaluation_points_vtk(points,filename=save_loc+"/"+str(rotor.tag)+"_eval_pts.vtk", time_step=time_step)                    
                    
                if verbose:
                    print("\nStoring VTKs...")
                
                Results = Data()
                if vlm_results is not None:
                    Results.vlm_results = vlm_results
                
                save_vehicle_vtks(vehicle, None, Results, time_step=time_step,save_loc=save_loc)  
    
                if wingPresent:
                    # store wing collocation points
                    points=Data()
                    points.XC = VD_wing.XC
                    points.YC = VD_wing.YC
                    points.ZC = VD_wing.ZC
                    save_evaluation_points_vtk(points,filename=save_loc+"wing_collocation_pts.vtk",time_step=time_step)
                
                
                # save CG Location
                CG_Point = Data()
                CG_Point.XC = np.array([vehicle.mass_properties.center_of_gravity[0][0]])
                CG_Point.YC = np.array([vehicle.mass_properties.center_of_gravity[0][1]])
                CG_Point.ZC = np.array([vehicle.mass_properties.center_of_gravity[0][2]])
                save_evaluation_points_vtk(CG_Point, filename = save_loc+"/CG.vtk", time_step=time_step)

                time_step +=1                
    return