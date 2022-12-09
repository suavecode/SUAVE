## @ingroup Methods-Propulsion
# set_optimized_rotor_planform.py 
#
# Created: Feb 2022, M. Clarke

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  
# SUAVE Imports 
import SUAVE 
from SUAVE.Core                                                                              import Units 
from SUAVE.Analyses.Mission.Segments.Segment                                                 import Segment 
from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_mid_fidelity                       import propeller_mid_fidelity 
from SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics                                 import Aerodynamics  

# Python package imports   
import numpy as np 
import scipy as sp  


def set_optimized_rotor_planform(rotor,optimization_problem):
    """ Append parameters of optimized prop-rotor to input prop-rotor
          
          Inputs:  
             prop-rotor                - prop-rotor data structure                   [None]
             optimization_problem      - data struction of optimized parameters      [None]
             
          Outputs: 
             prop-rotor                - prop-rotor data structure                   [None]
              
          Assumptions: 
             1) Default noise measurements are taken 135 degrees from prop-rotor plane 
        
          Source:
             None
    """    
    results                         = optimization_problem.results
    optimal_hover_rotor_network     = optimization_problem.vehicle_configurations.hover.networks.battery_propeller.propellers
    optimal_hover_rotor             = optimal_hover_rotor_network.rotor  
    alpha                           = optimal_hover_rotor.optimization_parameters.multiobjective_aeroacoustic_weight 
    theta                           = optimal_hover_rotor.optimization_parameters.noise_evaluation_angle   
    rotor.chord_distribution        = optimal_hover_rotor.chord_distribution
    rotor.twist_distribution        = optimal_hover_rotor.twist_distribution   
    
    if rotor.hover.design_power == None: 
        rotor.hover.design_power = results.hover.power 
    
    if rotor.hover.design_thrust == None: 
        rotor.hover.design_thrust = results.hover.thrust   
        
    rotor.hover.design_torque              = results.hover.torque  
    rotor.hover.design_angular_velocity    = results.hover.omega 
    rotor.hover.design_performance         = results.hover.full_results
    rotor.hover.design_Cl                  = results.hover.mean_CL 
    rotor.hover.design_thrust_coefficient  = results.hover.thurst_c
    rotor.hover.design_power_coefficient   = results.hover.power_c
     

    rotor.OEI.design_thrust                = results.OEI.thrust    
    rotor.OEI.design_power                 = results.OEI.power 
    rotor.OEI.design_torque                = results.OEI.torque  
    rotor.OEI.design_angular_velocity      = results.OEI.omega   
    
    if alpha == 1.0:
        
        # Calculate atmospheric properties
        V_hover              = rotor.hover.design_freestream_velocity  
        alt_hover            = rotor.hover.design_altitude 
        atmosphere           = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmo_data            = atmosphere.compute_values(alt_hover)    
    
        # microphone locations
        S_hover             = np.maximum(alt_hover,20*Units.feet)  
        mic_positions_hover = np.array([[0.0 , S_hover*np.sin(theta)  ,S_hover*np.cos(theta)]])  
    
        # Define run conditions 
        ctrl_pts                                         = 1   
        rotor.inputs.omega                               = np.array([[rotor.hover.design_angular_velocity]])
        rotor.orientation_euler_angles                   = [0.0,np.pi,0.0] 
        conditions                                       = Aerodynamics()   
        conditions.freestream.update(atmo_data)         
        conditions.frames.inertial.velocity_vector       = np.array([[0, 0. ,V_hover]]) 
        conditions.propulsion.throttle                   = np.ones((ctrl_pts,1))*1.0
        conditions.frames.body.transform_to_inertial     = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., -1.]]])  
    
        # Run rotor model 
        thrust_hover,_, power_hover, Cp_hover,noise_data_hover, _ = rotor.spin(conditions) 
    
        # Set up noise model
        conditions.noise.total_microphone_locations      = np.repeat(mic_positions_hover[ np.newaxis,:,: ],1,axis=0)
        conditions.aerodynamics.angle_of_attack          = np.ones((ctrl_pts,1))* 0. * Units.degrees 
        segment                                          = Segment() 
        segment.state.conditions                         = conditions
        segment.state.conditions.expand_rows(ctrl_pts)
        noise                                            = SUAVE.Analyses.Noise.Fidelity_One() 
        settings                                         = noise.settings   
        num_mic                                          = len(conditions.noise.total_microphone_locations[0])  
        conditions.noise.number_of_microphones           = num_mic
 
        propeller_noise       = propeller_mid_fidelity(optimal_hover_rotor_network,noise_data_hover,segment,settings)
        
        rotor.hover.design_noise   = propeller_noise
        rotor.hover.design_SPL_dBA = np.mean(propeller_noise.SPL_dBA)    
    else:
        
        rotor.hover.design_noise               = results.hover.noise_data
        rotor.hover.design_SPL_dBA             = results.hover.mean_SPL

    
    if optimization_problem.prop_rotor: 
    
        if rotor.cruise.design_power == None: 
            rotor.cruise.design_power = results.cruise.power 
        
        if rotor.cruise.design_thrust == None: 
            rotor.cruise.design_thrust = results.cruise.thrust            
    
        optimal_cruise_rotor_network            = optimization_problem.vehicle_configurations.cruise.networks.battery_propeller.propellers 
        optimal_cruise_rotor                    = optimal_cruise_rotor_network.rotor  
        
        rotor.cruise.design_torque              = results.cruise.torque  
        rotor.cruise.design_angular_velocity    = results.cruise.omega 
        rotor.cruise.design_performance         = results.cruise.full_results
        rotor.cruise.design_Cl                  = results.cruise.mean_CL 
        rotor.cruise.design_thrust_coefficient  = results.cruise.thurst_c
        rotor.cruise.design_power_coefficient   = results.cruise.power_c 
        rotor.cruise.collective_pitch           = optimal_cruise_rotor
        
        if alpha == 1.0:    
            # unpack prop-rotor properties     
            V_cruise             = rotor.cruise.design_freestream_velocity   
            alt_cruise           = rotor.cruise.design_altitude  
            atmo_data            = atmosphere.compute_values(alt_cruise)      
        
            # Microhpone locations  
            S_cruise             = np.maximum(alt_cruise, 20*Units.feet) 
            mic_positions_cruise = np.array([[1E-3,S_cruise*np.sin(theta)  ,S_cruise*np.cos(theta)]])   
        
            # Define run conditions 
            rotor.inputs.omega                               = np.array([[rotor.cruise.design_angular_velocity]])
            rotor.orientation_euler_angles                   = [0.0,0.0,0.0] 
            conditions                                       = Aerodynamics()   
            conditions.freestream.update(atmo_data)         
            conditions.frames.inertial.velocity_vector       = np.array([[V_cruise, 0. ,0.]])
            conditions.propulsion.throttle                   = np.ones((ctrl_pts,1))*1.0
            conditions.frames.body.transform_to_inertial     = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])
        
            # Run Propeller model 
            thrust_cruise ,_, power_cruise, Cp_cruise  , noise_data_cruise , etap_cruise  = rotor.spin(conditions)  
        
            # Set up noise model
            conditions.noise.total_microphone_locations      = np.repeat(mic_positions_cruise[ np.newaxis,:,: ],1,axis=0)
            conditions.aerodynamics.angle_of_attack          = np.ones((ctrl_pts,1))* 0. * Units.degrees 
            segment                                          = Segment() 
            segment.state.conditions                         = conditions
            segment.state.conditions.expand_rows(ctrl_pts)   
            noise                                            = SUAVE.Analyses.Noise.Fidelity_One() 
            settings                                         = noise.settings   
            num_mic                                          = len(conditions.noise.total_microphone_locations[0])  
            conditions.noise.number_of_microphones           = num_mic   

            propeller_noise             = propeller_mid_fidelity(optimal_cruise_rotor_network,noise_data_cruise,segment,settings)
        
            rotor.cruise.design_noise   = propeller_noise
            rotor.cruise.design_SPL_dBA = np.mean(propeller_noise.SPL_dBA)    
        else: 
            rotor.cruise.design_noise   = results.cruise.noise_data
            rotor.cruise.design_SPL_dBA = results.cruise.mean_SPL
             
    rotor.max_thickness_distribution        = optimal_hover_rotor.max_thickness_distribution  
    rotor.radius_distribution               = optimal_hover_rotor.radius_distribution         
    rotor.number_of_blades                  = optimal_hover_rotor.number_of_blades               
    rotor.mid_chord_alignment               = optimal_hover_rotor.mid_chord_alignment         
    rotor.thickness_to_chord                = optimal_hover_rotor.thickness_to_chord          
    rotor.blade_solidity                    = optimal_hover_rotor.blade_solidity   
    
    return rotor 