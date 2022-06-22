## @ingroup Input_Output-VTK
# save_vehicle_vtks.py
#
# Created:    Jun 2021, R. Erhard
# Modified:
#

#----------------------------
# Imports
#----------------------------

from SUAVE.Input_Output.VTK.save_wing_vtk import save_wing_vtk
from SUAVE.Input_Output.VTK.save_prop_vtk import save_prop_vtk
from SUAVE.Input_Output.VTK.save_prop_wake_vtk import save_prop_wake_vtk
from SUAVE.Input_Output.VTK.save_fuselage_vtk import save_fuselage_vtk
from SUAVE.Input_Output.VTK.save_nacelle_vtk import save_nacelle_vtk
from SUAVE.Input_Output.VTK.save_vortex_distribution_vtk import save_vortex_distribution_vtk

from SUAVE.Analyses.Aerodynamics import Vortex_Lattice

from SUAVE.Core import Data
import numpy as np
import os

## @ingroup Input_Output-VTK
def save_vehicle_vtks(vehicle, conditions=None, Results=Data(), 
                      time_step=0,origin_offset=np.array([0.,0.,0.]),VLM_settings=None, 
                      prop_filename="propeller.vtk", rot_filename="rotor.vtk",
                      wake_filename="prop_wake.vtk", wing_vlm_filename="wing_vlm_horseshoes.vtk",wing_filename="wing_vlm.vtk", 
                      fuselage_filename="fuselage.vtk", nacelle_filename="nacelle.vtk", save_loc=None):
    """
    Saves SUAVE vehicle components as VTK files in legacy format.

    Inputs:
       vehicle                Data structure of SUAVE vehicle                    [Unitless]
       settings               Settings for aerodynamic analysis                  [Unitless]
       Results                Data structure of wing and propeller results       [Unitless]
       time_step              Simulation time step                               [Unitless]
       prop_filename          Name of vtk file to save                           [String]
       rot_filename           Name of vtk file to save                           [String]
       wake_filename          Name of vtk file to save                           [String]
       wing_filename          Name of vtk file to save                           [String]
       fuselage_filename      Name of vtk file to save                           [String]
       save_loc               Location at which to save vtk files                [String]

    Outputs:
       N/A

    Properties Used:
       N/A

    Assumptions:
       Quad cell structures for mesh

    Source:
       None

    """
    if (save_loc is not None) and (not os.path.exists(save_loc)):
        os.makedirs(save_loc)
        print("Directory "+save_loc+" created.") 
        
    if VLM_settings == None:
        VLM_settings = Vortex_Lattice().settings
        VLM_settings.number_spanwise_vortices  = 25
        VLM_settings.number_chordwise_vortices = 5
        VLM_settings.spanwise_cosine_spacing   = False
        VLM_settings.model_fuselage            = False
        VLM_settings.model_nacelle             = False


    #---------------------------
    # Save propellers and rotors to vtk
    #---------------------------
    for network in vehicle.networks:
        try:
            print("Attempting to save propeller.")
            propellers = network.propellers
            try:
                n_props = int(network.number_of_propeller_engines)
            except:
                n_props   = int(network.number_of_engines)
        except:
            print("No propellers.")
            n_props = 0

        if n_props>0:
            for i in range(n_props):
                propi = propellers[list(propellers.keys())[i]]
                
                propi.inputs.y_axis_rotation = network.y_axis_rotation
                
                start_angle = propi.start_angle
                Na = propi.number_azimuthal_stations
                angles = np.linspace(0,2*np.pi,Na+1)[:-1]
                start_angle_idx = np.where(np.isclose(abs(start_angle),angles))[0][0]
                
                
                # save the ith propeller
                if save_loc==None:
                    filename = prop_filename
                else:
                    filename = save_loc + prop_filename
                    
                sep  = filename.rfind('.')
                file = filename[0:sep]+str(i)+filename[sep:]

                save_prop_vtk(propi, file, Results, time_step, origin_offset)
                
                try:
                    # check if rotor has wake present
                    gamma = propi.Wake.vortex_distribution.reshaped_wake.GAMMA[start_angle_idx,:,:,:,:]
                    wVD = propi.Wake.vortex_distribution.reshaped_wake
                    wake_present = True
                except:
                    wake_present = False
                    pass
                
                if wake_present:
                    #---------------------------
                    # Save propeller wake to vtk
                    #---------------------------
                    # save the wake of the ith propeller
                    if save_loc ==None:
                        filename = wake_filename
                    else:
                        filename = save_loc + wake_filename
                    sep  = filename.rfind('.')
                    file = filename[0:sep]+str(i)+"_t."+str(time_step)+filename[sep:]
                
                    Results['prop_outputs'] = propi.outputs
                    
                    # save prop wake
                    save_prop_wake_vtk(wVD, gamma, file, Results,start_angle_idx,origin_offset,rot=propi.rotation) 
                
                    
        try:
            print("Attempting to save rotor.")
            lift_rotors = network.lift_rotors
            if network.number_of_lift_rotor_engines is not None:
                n_rots = int(network.number_of_lift_rotor_engines)
            else:
                n_rots = 0
        except:
            print("No lift rotors.")
            n_rots = 0


        if n_rots > 0:
            for i in range(n_rots):
                roti = lift_rotors[list(lift_rotors.keys())[i]]
                

                start_angle = roti.start_angle
                Na = roti.number_azimuthal_stations
                angles = np.linspace(0,2*np.pi,Na+1)[1:]
                start_angle_idx = np.where(start_angle==angles)
                
                
                # save the ith rotor
                if save_loc ==None:
                    filename = prop_filename
                else:
                    filename = save_loc + rot_filename
                sep  = filename.rfind('.')
                file = filename[0:sep]+str(i)+filename[sep:]

                save_prop_vtk(roti, file, Results,i,time_step, origin_offset)
                
                try:
                    # check if rotor has wake present
                    gamma = roti.vortex_distribution.reshaped_wake.GAMMA
                    wVD = propi.vortex_distribution.reshaped_wake
                    wake_present = True
                except:
                    wake_present = False
                    pass
                
                if wake_present:                    
                    #---------------------------
                    # Save propeller wake to vtk
                    #---------------------------
                    # save the wake of the ith propeller
                    if save_loc ==None:
                        filename = wake_filename
                    else:
                        filename = save_loc + wake_filename
                    sep  = filename.rfind('.')
                    file = filename[0:sep]+str(i)+"_t."+str(time_step)+filename[sep:]
                
                    roti_key = list(Results['all_prop_outputs'].keys())[i]
                    Results['prop_outputs'] = Results['all_prop_outputs'][roti_key]
                    
                    # save prop wake
                    save_prop_wake_vtk(wVD, gamma, file, Results,origin_offset,rot=roti.rotation)      

    
    #---------------------------
    # Save wing results to vtk
    #---------------------------
    wing_names = list(vehicle.wings.keys())
    n_wings    = len(wing_names)
    for i in range(n_wings):
        if save_loc ==None:
            filename = wing_filename
            filename2 = wing_vlm_filename
        else:
            filename = save_loc + wing_filename
            filename2 = save_loc + wing_vlm_filename


        sep  = filename.rfind('.')
        file = filename[0:sep]+str(wing_names[i])+filename[sep:]
        file2 = filename2[0:sep]+str(wing_names[i])+filename2[sep:]
        save_wing_vtk(vehicle, vehicle.wings[wing_names[i]], VLM_settings, file, Results,time_step,origin_offset)
        
        if conditions != None:
            # evaluate vortex strengths and same vortex distribution
            VLM_outputs   = VLM(conditions, VLM_settings, vehicle) 
            gamma         = VLM_outputs.gamma
            save_vortex_distribution_vtk(vehicle,conditions,VD,gamma,vehicle.wings[wing_names[i]], file2, time_step)

    #------------------------------
    # Save fuselage results to vtk
    #------------------------------
    fuselages    = list(vehicle.fuselages.keys())
    n_fuselage   = len(fuselages)
    for i in range(n_fuselage):
        if save_loc ==None:
            filename = fuselage_filename
        else:
            filename = save_loc + fuselage_filename
        sep  = filename.rfind('.')
        file = filename[0:sep]+str(i)+"_t"+str(time_step)+filename[sep:]

        save_fuselage_vtk(vehicle, file, Results, origin_offset)

    
    #------------------------------
    # Save nacelles to vtk
    #------------------------------
    nacelles    = vehicle.nacelles
    for i, nacelle in enumerate(nacelles):
        if save_loc ==None:
            filename = nacelle_filename
        else:
            filename = save_loc + nacelle_filename
        sep  = filename.rfind('.')
        file = filename[0:sep]+str(i)+"_t"+str(time_step)+filename[sep:]

        save_nacelle_vtk(nacelle, file, Results, origin_offset)
    return
