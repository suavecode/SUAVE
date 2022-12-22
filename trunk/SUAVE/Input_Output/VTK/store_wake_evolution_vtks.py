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
def store_wake_evolution_vtks(wake,rotor,save_loc=None):
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
    rotor_outputs = rotor.outputs
    Na            = rotor_outputs.number_azimuthal_stations
    omega         = rotor_outputs.omega                 
    VD            = rotor.vortex_distribution
    
    # Get start angle of rotor
    azi   = np.linspace(0,2*np.pi,Na+1)[:-1]
    ito   = wake.wake_settings.initial_timestep_offset
    dt    = (azi[1]-azi[0])/omega[0][0]
    t0    = dt*ito
    
    # --------------------------------------------------------------------------------------------------------------
    #    Store VTKs after wake is generated
    # --------------------------------------------------------------------------------------------------------------      
    if save_loc == None:
        print("No location set for saving wake evolution VTKs!")
        pass
    else:
        # after converged, store vtks for final wake shape for each of Na starting positions
        for i in range(Na):
            # increment blade angle to new azimuthal position
            blade_angle       = (omega[0][0]*t0 + i*(2*np.pi/(Na))) * rotor.rotation  # Positive rotation, positive blade angle
            rotor.start_angle = blade_angle

            print("\nStoring VTKs...")
            
            # create dummy vehicle
            vehicle = SUAVE.Vehicle()
            net     = SUAVE.Components.Energy.Networks.Battery_Rotor()
            net.y_axis_rotation = rotor.inputs.y_axis_rotation
            net.number_of_engines  = 1
            net.propellers.append(rotor)
            vehicle.append_component(net) 

            save_vehicle_vtks(vehicle, Results=Data(), time_step=i,save_loc=save_loc)  

            Yb   = wake.vortex_distribution.reshaped_wake.Yblades_cp[i,0,0,:,0] 
            Zb   = wake.vortex_distribution.reshaped_wake.Zblades_cp[i,0,0,:,0] 
            Xb   = wake.vortex_distribution.reshaped_wake.Xblades_cp[i,0,0,:,0] 

            VD.YC = (Yb[1:] + Yb[:-1])/2
            VD.ZC = (Zb[1:] + Zb[:-1])/2
            VD.XC = (Xb[1:] + Xb[:-1])/2

            points = Data()
            points.XC = VD.XC
            points.YC = VD.YC
            points.ZC = VD.ZC
            points.induced_velocities = Data()
            points.induced_velocities.va = rotor_outputs.disc_axial_induced_velocity[0,:,i]
            points.induced_velocities.vt = rotor_outputs.disc_tangential_induced_velocity[0,:,i]
            save_evaluation_points_vtk(points,filename=save_loc+"/eval_pts.vtk", time_step=i)
    return