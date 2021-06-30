# VTK_Test.py
# 
# Created:  Jun 2021, R. Erhard
# Modified: 

""" generates vtk files for X57 aircraft 
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data, Units
import SUAVE
import pylab as plt
import numpy as np
import copy
import time
import sys

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.VLM import VLM
from SUAVE.Input_Output.VTK.save_vehicle_vtk import save_vehicle_vtk
sys.path.append('../Vehicles') 
from X57_Maxwell import vehicle_setup 


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    """
    generates vtk files for the X57 aircraft
    """
    # Set the vehicle and single point conditions
    x57             = vehicle_setup()
    inputs          = set_inputs()
    state, settings = set_conditions(inputs, x57)
    
    # Run a single point analysis 
    Results = run_simulation(x57, state.conditions, settings, timesteps=1)
    
    # Save the vehicle vtk files
    for key in Results.keys():
        save_vehicle_vtk(x57, settings, Results[key])
    
    return 


def set_inputs():
    inputs = Data()
    inputs.omega = 200
    inputs.alpha = 3 * Units.deg
    inputs.altitude = 2500. * Units.meter
    inputs.velocity = 175. * Units.mph
    inputs.rotation = [1]
    
    return inputs

def set_conditions(inputs,vehicle):
    alt      = inputs.altitude
    Vinf     = inputs.velocity
    omega    = inputs.omega 
    alpha    = inputs.alpha
    rotation = inputs.rotation
    
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data  = atmosphere.compute_values(altitude=alt * Units.ft)
    rho        = atmo_data.density
    mu         = atmo_data.dynamic_viscosity
    T          = atmo_data.temperature
    P          = atmo_data.pressure
    a          = atmo_data.speed_of_sound
    Mach       = Vinf/a    
    
    AoA               = alpha*np.ones((1,1))  
    state            = SUAVE.Analyses.Mission.Segments.Conditions.State()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()    
    
    state.conditions.freestream.mach_number             = Mach  * np.ones_like(AoA) 
    state.conditions.freestream.density                 = rho   * np.ones_like(AoA) 
    state.conditions.freestream.dynamic_viscosity       = mu    * np.ones_like(AoA) 
    state.conditions.freestream.temperature             = T     * np.ones_like(AoA) 
    state.conditions.freestream.pressure                = P     * np.ones_like(AoA) 
    state.conditions.freestream.reynolds_number         = rho*a*Mach/mu      * np.ones_like(AoA)
    state.conditions.freestream.velocity                = a*Mach* np.ones_like(AoA) 
    state.conditions.aerodynamics.angle_of_attack       = AoA  
    state.conditions.frames                             = Data()  
    state.conditions.frames.inertial                    = Data()  
    state.conditions.frames.body                        = Data()  
    #state.conditions.use_Blade_Element_Theory           = False
    state.conditions.frames.body.transform_to_inertial  = np.array([[[1., 0., 0],[0., 1., 0.],[0., 0., 1.]]]) 
    state.conditions.propulsion.throttle                = np.ones((1,1))
    
    velocity_vector                                     = np.array([[Vinf, 0. ,0.]])
    state.conditions.frames.inertial.velocity_vector    = np.tile(velocity_vector,(1,1))     
    
    
    for propulsor in vehicle.propulsors:
        try:
            prop = propulsor.propeller
            prop.inputs.omega    = np.ones((1,1)) * omega
            prop.inputs.rotation = rotation
        except:
            print("No propeller. Looking for rotor.")
        try:
            rot = propulsor.propeller
            rot.inputs.omega    = np.ones((1,1)) * omega
            rot.inputs.rotation = rotation 
        except:
            print("No rotor specified.")
    
    settings = Data()
    settings.use_surrogate                         = False    
    settings.propeller_wake_model                  = True    
    settings.use_bemt_wake_model                   = False 
    settings.number_spanwise_vortices              = 50
    settings.number_chordwise_vortices             = 10 
    settings.prop_wake_model                       = 'Stream_Tube'  
    settings.initial_timestep_offset               = 0.0   
    settings.wake_development_time                 = 0.05 # sensitive
    settings.number_of_wake_timesteps              = 30#72
    settings.spanwise_cosine_spacing               = True 
    settings.leading_edge_suction_multiplier       = 1
    settings.model_fuselage                        = True  
    
    
    return state, settings

#def set_conditions(inputs, vehicle, tiltwing=False):
    ## =========================================================================================================
    ## Conditions set in Veldhuis's Thesis for PROWIM Experimental Test
    ## =========================================================================================================
    #rho      = 1.225     # 1.20857  
    #mu       = 1.81e-05  # 1.78857e-05
    #T        = 288.
    #P        = 99915.9   
    #a        = np.sqrt(1.4*287.058*T)  
    
    #Vinf     = 175. * Units['mph']
    #Mach     = Vinf/a    
    
    #omega    = 198.30015463 #n*2*np.pi
    
    #return state, settings

def run_simulation(vehicle, conditions, settings, timesteps):
    ti = time.time()
    # =========================================================================================================
    # Run Propeller model 
    # =========================================================================================================
    propeller                  = vehicle.propulsors.battery_propeller.propeller
    F, Q, P, Cp, outputs, etap = propeller.spin(conditions)
    propeller.outputs          = outputs
    
    # =========================================================================================================
    # Run VLM with DVE Fixed Wake Model  
    # =========================================================================================================
    # Create Empty Arrays for Storing Results 
    Results = Data()   
    dx_wake = 0
    for init_timestep_offset in range(timesteps):
        settings.initial_timestep_offset = init_timestep_offset
        CL, CDi, CM, CL_wing, CDi_wing, cl_y , cdi_y , AoAi , CP ,g =  VLM(conditions,settings,vehicle)
        VD = vehicle.vortex_distribution  
        Results['timestep_'+str(init_timestep_offset)] = {'CL_DVE':CL, 
                                                              'CDi_DVE':CDi,
                                                              'CM_DVE': CM,
                                                              'CL_wing_DVE': CL_wing,
                                                              'CDi_wing_DVE': CDi_wing,
                                                              'cl_y_DVE':cl_y ,
                                                              'cdi_y_DVE': cdi_y ,
                                                              'CP_DVE':CP ,
                                                              #'Velocities':V_distribution,
                                                              'VD':VD,
                                                              'prop_outputs':outputs}   
        
        ## Update timing
        #dt = settings.wake_development_time/settings.number_of_wake_timesteps
        #t0 = dt*init_timestep_offset
        
        
        ## Update streamwise position of vehicle and wake for simulation:
        #if init_timestep_offset == 0:
            #vehicle_sim = copy.deepcopy(vehicle)
        #vehicle_sim.vortex_distribution.Wake = copy.deepcopy(vehicle.vortex_distribution.Wake)
        #dx      = dt*conditions.freestream.velocity[0][0]
        #dx_wake = dx_wake + dx
        
        
        
        ## Update wing and propeller orientation
        #vehicle_sim = update_vehicle_orientation(vehicle_sim, dx, t0)
        #vehicle_sim = update_wake(vehicle_sim, dx_wake, t0)
        
        #Gprops = prop_points(vehicle_sim.propulsors.battery_propeller)
        
        ## Save vtk files for the wing and propellers
        #save_vtks(vehicle_sim, Results['timestep_'+str(init_timestep_offset)], Gprops,settings,
                  #prop_filename="prowim_prop."+str(init_timestep_offset)+".vtk",
                  #wing_filename="prowim_wing_vlm."+str(init_timestep_offset)+".vtk",
                  #wake_filename="prowim_prop_wake."+str(init_timestep_offset)+".vtk", 
                  #save_loc="/Users/rerha/Desktop/PROWIM/PROWIM_SUAVE/prowim_blownwing_alpha4/")         

     
    # =========================================================================================================
    # 2D Plots of results
    # =========================================================================================================   
    #plot_spanwise_loading(Results,vehicle,timesteps)           
    
    tf = time.time()
    print ('Time taken: ' + str((tf-ti)/60) + ' mins')   
    
    
    return Results



if __name__ == '__main__': 
    main()    
    plt.show()