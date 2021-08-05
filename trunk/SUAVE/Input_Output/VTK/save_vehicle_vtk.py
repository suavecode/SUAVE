## @ingroup Input_Output-VTK
# save_vehicle_vtks.py
# 
# Created:    Jun 2021, R. Erhard
# Modified: 
#  

#----------------------------
# Imports
#----------------------------
from SUAVE.Core import Data
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_wing_vortex_distribution  import generate_wing_vortex_distribution
from SUAVE.Input_Output.VTK.save_wing_vtk import save_wing_vtk
from SUAVE.Input_Output.VTK.save_prop_vtk import save_prop_vtk
from SUAVE.Input_Output.VTK.save_prop_wake_vtk import save_prop_wake_vtk
from SUAVE.Input_Output.VTK.save_fuselage_vtk import save_fuselage_vtk


def save_vehicle_vtks(vehicle, settings, Results, time_step, prop_filename="propeller.vtk", rot_filename="rotor.vtk",
                     wake_filename="prop_wake.vtk", wing_filename="wing_vlm.vtk", fuselage_filename="fuselage.vtk", save_loc=None):
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
    # unpack vortex distribution 
    try:
        VD = vehicle.vortex_distribution 
    except:
        print("Simulation has not yet been run. Generating vortex distribution for geometry export.")
        settings = Data()
        settings.number_spanwise_vortices  = 25
        settings.number_chordwise_vortices = 5
        settings.spanwise_cosine_spacing   = False 
        settings.model_fuselage            = False
        VD = generate_wing_vortex_distribution(vehicle,settings) 
        vehicle.vortex_distribution = VD
    
    
    #---------------------------
    # Save propellers and rotors to vtk
    #---------------------------
    for network in vehicle.networks:
        try:
            print("Attempting to save propeller.")
            propeller = network.propeller
            try:
                n_props = int(network.number_of_propeller_engines)
            except:
                n_props   = int(network.number_of_engines)
        except:
            print("No propeller.")
            n_props = 0
            
        if n_props>0:
            for i in range(n_props):
                # save the ith propeller
                if save_loc ==None:
                    filename = prop_filename
                else:
                    filename = save_loc + prop_filename
                sep  = filename.find('.')
                file = filename[0:sep]+str(i)+filename[sep:]
            
                save_prop_vtk(propeller, file, Results,i, time_step) 
                
        try:
            print("Attempting to save rotor.")
            rotor = network.rotor
            try:
                n_rots = int(network.number_of_rotor_engines)
            except:
                n_rots = int(network.number_of_engines)    
        except:
            print("No rotor.") 
            n_rots = 0
            
        if n_rots > 0:
            for i in range(n_rots):
                # save the ith rotor
                if save_loc ==None:
                    filename = prop_filename
                else:
                    filename = save_loc + rot_filename
                sep  = filename.find('.')
                file = filename[0:sep]+str(i)+filename[sep:]        
                
                save_prop_vtk(rotor, file, Results,i,time_step) 
                       
        
    
    #---------------------------
    # Save propeller wake to vtk
    #---------------------------
    try:
        n_wakes = len(VD.Wake.XA1[:,0,0,0])
        for i in range(n_wakes):
            # save the wake of the ith propeller
            if save_loc ==None:
                filename = wake_filename
            else:            
                filename = save_loc + wake_filename 
            sep  = filename.find('.')
            file = filename[0:sep]+str(i)+"_t"+str(time_step)+filename[sep:]
            save_prop_wake_vtk(VD.Wake, file, Results,i) 
    except:
        print("Wake simulation has not yet been run. No propeller wakes generated.")
    
    #---------------------------
    # Save wing results to vtk
    #---------------------------
    wing_names = list(vehicle.wings.keys())
    n_wings    = len(wing_names)
    for i in range(n_wings):
        if save_loc ==None:
            filename = wing_filename
        else:           
            filename = save_loc + wing_filename 
        sep  = filename.find('.')
        file = filename[0:sep]+str(wing_names[i])+filename[sep:]
        save_wing_vtk(vehicle, vehicle.wings[wing_names[i]], settings, file, Results,time_step)
        
        
    #------------------------------
    # Save fuselage results to vtk
    #------------------------------
    n_fuselage    = len(vehicle.fuselages.keys())
    for i in range(n_fuselage):
        if save_loc ==None:
            filename = fuselage_filename
        else:           
            filename = save_loc + fuselage_filename 
        sep  = filename.find('.')
        file = filename[0:sep]+str(i)+"_t"+str(time_step)+filename[sep:]
        
        save_fuselage_vtk(vehicle, file, Results)
    
    return

