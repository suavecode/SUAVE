from SUAVE.Core import Data
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_wing_vortex_distribution  import generate_wing_vortex_distribution
from SUAVE.Time_Accurate.Simulations.save_wing_vtk import save_wing_vtk
from SUAVE.Time_Accurate.Simulations.save_prop_vtk import save_prop_vtk
from SUAVE.Time_Accurate.Simulations.save_prop_wake_vtk import save_prop_wake_vtk
from SUAVE.Time_Accurate.Simulations.save_fuselage_vtk import save_fuselage_vtk


def save_vehicle_vtk(vehicle, settings, Results, Gprops, prop_filename="prop.vtk",wake_filename="prop_wake.vtk", 
              wing_filename="wing_vlm.vtk", fuselage_filename="fuselage.vtk", save_loc=None, tiltwing=False):
    """
    
    """
    # unpack vortex distribution 
    try:
        VD = vehicle.vortex_distribution 
    except:
        settings = Data()
        settings.number_spanwise_vortices  = 25
        settings.number_chordwise_vortices = 5
        settings.spanwise_cosine_spacing   = False 
        settings.model_fuselage            = False
        VD = generate_wing_vortex_distribution(vehicle,settings) 
        vehicle.vortex_distribution = VD
    
    
    #---------------------------
    # Save propeller to vtk
    #---------------------------
    for propulsor in vehicle.propulsors:
        try:
            propeller = propulsor.propeller
            n_props = int(propulsor.number_of_engines)
            for i in range(n_props):
                # save the ith propeller
                filename = save_loc + prop_filename
                sep  = filename.find('.')
                file = filename[0:sep]+str(i)+filename[sep:]        
                
                save_prop_vtk(propeller, file, Results,i, Gprops) 
        except:
            print("No propeller.")
        try:
            rotor = propulsor.rotor
            n_rots = int(propulsor.number_of_engines)
            for i in range(n_rots):
                # save the ith rotor
                filename = save_loc + prop_filename
                sep  = filename.find('.')
                file = filename[0:sep]+str(i)+filename[sep:]        
                
                save_prop_vtk(rotor, file, Results,i, Gprops) 
        except:
            print("No rotor.")                
        
           

    
    #---------------------------
    # Save propeller wake to vtk
    #---------------------------
    try:
        n_wakes = len(VD.Wake.XA1[:,0,0,0])
        for i in range(n_wakes):
            # save the wake of the ith propeller
            filename = save_loc + wake_filename 
            sep  = filename.find('.')
            file = filename[0:sep]+str(i)+filename[sep:]
            save_prop_wake_vtk(VD, file, Results,i) 
    except:
        print("Wake simulation has not yet been run. No propeller wakes generated.")
    
    #---------------------------
    # Save wing results to vtk
    #---------------------------
    wing_names = list(vehicle.wings.keys())
    n_wings    = len(wing_names)
    for i in range(n_wings):
        filename = save_loc + wing_filename 
        sep  = filename.find('.')
        file = filename[0:sep]+str(wing_names[i])+filename[sep:]
        save_wing_vtk(vehicle, vehicle.wings[wing_names[i]], settings, file, Results)
        
        
    #------------------------------
    # Save fuselage results to vtk
    #------------------------------
    n_fuselage    = len(vehicle.fuselages.keys())
    for i in range(n_fuselage):
        filename = save_loc + fuselage_filename 
        sep  = filename.find('.')
        file = filename[0:sep]+str(i)+filename[sep:]
        save_fuselage_vtk(vehicle, settings, file, Results)
            
    
    
    return