## @ingroup Methods-Propulsion-Rotor_Design
# procedure_setup.py 
#
# Created: Feb 2022, M. Clarke

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  
# SUAVE Imports 
import SUAVE 
from SUAVE.Core                                                                                import Units 
from SUAVE.Analyses.Mission.Segments.Segment                                                   import Segment 
from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_mid_fidelity                         import propeller_mid_fidelity 
from SUAVE.Analyses.Process                                                                    import Process   

# Python package imports  
from numpy import linalg as LA  
import numpy as np 
import scipy as sp  

## @ingroup Methods-Propulsion-Rotor_Design 
def procedure_setup(): 
    
    # size the base config
    procedure = Process()
    
    # mofify blade geometry 
    procedure.modify_rotor = modify_blade_geometry
    
    # Run the rotor in hover
    procedure.hover       = run_rotor_hover
    procedure.hover_noise = run_hover_noise
     
    # Run the rotor in the OEI condition
    procedure.OEI         = run_rotor_OEI 

     # Run the rotor in cruise
    procedure.run_rotor_cruise = run_rotor_cruise
    procedure.cruise_noise     = run_cruise_noise 

    # post process the results
    procedure.post_process = post_process
        
    return procedure


# ----------------------------------------------------------------------
# Update blade geometry 
# ---------------------------------------------------------------------- 
def modify_blade_geometry(nexus): 
    """ Modifies geometry of prop-rotor blade 
          
          Inputs:  
             nexus     - SUAVE optmization framework with prop-rotor blade data structure [None]
              
          Outputs:   
             procedure - optimization methodology                                         [None]
              
          Assumptions: 
             N/A 
        
          Source:
             None
    """        
 
    # Pull out the vehicles

    vehicle_hover     = nexus.vehicle_configurations.hover
    rotor_hover       = vehicle_hover.networks.battery_propeller.propellers.rotor
    vehicle_oei       = nexus.vehicle_configurations.oei
    rotor_oei         = vehicle_oei.networks.battery_propeller.propellers.rotor    
    
    if nexus.prop_rotor:  
        vehicle_cruise    = nexus.vehicle_configurations.cruise 
        rotor_cruise      = vehicle_cruise.networks.battery_propeller.propellers.rotor 
        
    airfoils = rotor_hover.Airfoils      
    a_loc    = rotor_hover.airfoil_polar_stations   
        
    # Update geometry of blade
    R       = rotor_hover.tip_radius     
    B       = rotor_hover.number_of_blades 
    r       = rotor_hover.radius_distribution
    c       = updated_blade_geometry(rotor_hover.radius_distribution/rotor_hover.tip_radius ,rotor_hover.chord_r,rotor_hover.chord_p,rotor_hover.chord_q,rotor_hover.chord_t)     
    beta    = updated_blade_geometry(rotor_hover.radius_distribution/rotor_hover.tip_radius ,rotor_hover.twist_r,rotor_hover.twist_p,rotor_hover.twist_q,rotor_hover.twist_t)   
    
    # compute max thickness distribution   
    blade_area    = sp.integrate.cumtrapz(B*c, r-r[0])
    sigma         = blade_area[-1]/(np.pi*R**2)      
    t_max         = np.zeros(len(c))    
    t_c           = np.zeros(len(c))       
    t_max  = np.zeros(len(c))     
    if len(airfoils.keys())>0:
        for j,airfoil in enumerate(airfoils): 
            a_geo         = airfoil.geometry
            locs          = np.where(np.array(a_loc) == j )
            t_max[locs]   = a_geo.max_thickness*c[locs]   
     
    rotor_hover.chord_distribution          = c
    rotor_hover.twist_distribution          = beta  
    rotor_hover.mid_chord_alignment         = c/4. - c[0]/4.
    rotor_hover.max_thickness_distribution  = t_max 
    rotor_hover.thickness_to_chord          = t_c
    rotor_hover.blade_solidity              = sigma     
    vehicle_hover.store_diff() 
      

    rotor_oei.chord_distribution         = rotor_hover.chord_distribution
    rotor_oei.twist_distribution         = rotor_hover.twist_distribution
    rotor_oei.mid_chord_alignment        = rotor_hover.mid_chord_alignment  
    rotor_oei.max_thickness_distribution = rotor_hover.max_thickness_distribution
    rotor_oei.thickness_to_chord         = rotor_hover.thickness_to_chord 
    rotor_oei.blade_solidity             = rotor_hover.blade_solidity    
    vehicle_oei.store_diff()     
     
    if nexus.prop_rotor: 
        rotor_cruise.chord_distribution         = rotor_hover.chord_distribution
        rotor_cruise.twist_distribution         = rotor_hover.twist_distribution
        rotor_cruise.mid_chord_alignment        = rotor_hover.mid_chord_alignment  
        rotor_cruise.max_thickness_distribution = rotor_hover.max_thickness_distribution
        rotor_cruise.thickness_to_chord         = rotor_hover.thickness_to_chord 
        rotor_cruise.blade_solidity             = rotor_hover.blade_solidity     
        vehicle_cruise.store_diff()  
    
    return nexus    


# ----------------------------------------------------------------------
#   Update blade geometry 
# ---------------------------------------------------------------------- 
def updated_blade_geometry(chi,c_r,p,q,c_t):
    """ Computes planform function of twist and chord distributron using hyperparameters  
          
          Inputs:  
             chi - prop-rotor radius distribution [None]
             c_r - hyperparameter no. 1           [None]
             p   - hyperparameter no. 2           [None]
             q   - hyperparameter no. 3           [None]
             c_t - hyperparameter no. 4           [None]
                   
          Outputs:       
             x_lin  - function distribution       [None]
              
          Assumptions: 
             N/A 
        
          Source:
              Traub, Lance W., et al. "Effect of taper ratio at low reynolds number."
              Journal of Aircraft 52.3 (2015): 734-747.
              
    """           

    n       = np.linspace(len(chi)-1,0,len(chi))          
    theta_n = n*(np.pi/2)/len(chi)              
    y_n     = chi[-1]*np.cos(theta_n)          
    eta_n   = np.abs(y_n/chi[-1])            
    x_cos   = c_r*(1 - eta_n**p)**q + c_t*eta_n  
    x_lin   = np.interp(chi,eta_n, x_cos)  
    return x_lin 

# ----------------------------------------------------------------------
#   Run the Rotor Hover
# ---------------------------------------------------------------------- 
def run_rotor_OEI(nexus):
    
    # Unpack  
    rotor    = nexus.vehicle_configurations.oei.networks.battery_propeller.propellers.rotor
    
    # Setup Test conditions
    speed    = rotor.OEI.design_freestream_velocity 
    altitude = np.array([rotor.OEI.design_altitude]) 
    R        = rotor.tip_radius
    TM       = rotor.OEI.design_tip_mach   

    # Calculate the atmospheric properties
    atmosphere            = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere_conditions = atmosphere.compute_values(altitude)

    # Pack everything up
    conditions                                          = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    conditions.freestream.update(atmosphere_conditions)
    conditions.frames.inertial.velocity_vector          = np.array([[0.,0.,speed]])
    conditions.propulsion.throttle                      = np.array([[1.0]])
    conditions.frames.body.transform_to_inertial        = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., -1.]]]) 
    
    # Calculate the RPM
    tip_speed = atmosphere_conditions.speed_of_sound*TM
    omega     = tip_speed/R
    
    # Set the RPM
    rotor.inputs.omega = np.array(omega)    
    
    # Run the rotor
    F, Q, P, Cp, outputs, etap  = rotor.spin(conditions)
    
    # Pack the results
    nexus.results.OEI.thrust       = -F[0,2]  
    nexus.results.OEI.torque       = Q
    nexus.results.OEI.power        = P
    nexus.results.OEI.power_c      = Cp
    nexus.results.OEI.omega        = omega[0][0]
    nexus.results.OEI.thurst_c     = outputs.thrust_coefficient[0][0]
    nexus.results.OEI.full_results = outputs
    nexus.results.OEI.efficiency   = etap 
    nexus.results.OEI.conditions   = conditions 
    
    return nexus

# ----------------------------------------------------------------------
#   Run the Rotor Hover
# ----------------------------------------------------------------------  
def run_rotor_hover(nexus):
     
    # Unpack 
    rotor   = nexus.vehicle_configurations.hover.networks.battery_propeller.propellers.rotor   

    # Setup Test conditions
    speed    = rotor.hover.design_freestream_velocity 
    altitude = np.array([rotor.hover.design_altitude]) 
    R        = rotor.tip_radius
    TM       = rotor.hover.design_tip_mach  

    # Calculate the atmospheric properties
    atmosphere            = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere_conditions = atmosphere.compute_values(altitude)

    # Pack everything up
    conditions                                          = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    conditions.freestream.update(atmosphere_conditions)
    conditions.frames.inertial.velocity_vector          = np.array([[0.,0.,speed]])
    conditions.propulsion.throttle                      = np.array([[1.0]])
    conditions.frames.body.transform_to_inertial        = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., -1.]]]) 
    
    # Calculate the RPM
    tip_speed = atmosphere_conditions.speed_of_sound*TM
    omega       = tip_speed/R
    
    # Set the RPM
    rotor.inputs.omega = np.array(omega)    
    
    # Run the rotor
    F, Q, P, Cp, outputs, etap  = rotor.spin(conditions)
    
    # Pack the results
    nexus.results.hover.thrust           = -F[0,2]  
    nexus.results.hover.torque           = Q[0][0]
    nexus.results.hover.power            = P[0][0]
    nexus.results.hover.power_c          = Cp[0][0]
    nexus.results.hover.thurst_c         = outputs.thrust_coefficient[0][0]
    nexus.results.hover.omega            = omega[0][0]
    nexus.results.hover.max_sectional_cl = np.max(outputs.lift_coefficient[0]) 
    nexus.results.hover.mean_CL          = np.mean(outputs.lift_coefficient[0])
    nexus.results.hover.full_results     = outputs  

    # figure of merit    
    nexus.results.hover.figure_of_merit  = outputs.figure_of_merit[0][0] 
    
    nexus.results.hover.efficiency       = etap[0][0] 
    nexus.results.hover.conditions       = conditions
    return nexus


# ----------------------------------------------------------------------
#   Run the Rotor Cruise
# ----------------------------------------------------------------------  
def run_rotor_cruise(nexus):
 
    if nexus.prop_rotor:     
        rotor    = nexus.vehicle_configurations.cruise.networks.battery_propeller.propellers.rotor
        
        # Setup Test conditions
        speed    = rotor.cruise.design_freestream_velocity 
        altitude = np.array([rotor.cruise.design_altitude]) 
        R        = rotor.tip_radius
        TM       = rotor.cruise.design_tip_mach 
        
        # Calculate the atmospheric properties
        atmosphere            = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmosphere_conditions = atmosphere.compute_values(altitude)
    
        # Pack everything up
        conditions                                          = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
        conditions.freestream.update(atmosphere_conditions)
        conditions.frames.inertial.velocity_vector          = np.array([[speed,0.,0.]])
        conditions.propulsion.throttle                      = np.array([[1.0]])
        conditions.frames.body.transform_to_inertial        = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])
        
        # Calculate the RPM
        tip_speed = atmosphere_conditions.speed_of_sound*TM
        omega     = tip_speed/R
        
        # Set the RPM
        rotor.inputs.omega = np.array(omega)    
        
        # Run the rotor
        F, Q, P, Cp, outputs, etap  = rotor.spin(conditions)
        
        # Pack the results
        nexus.results.cruise.thrust           = F[0][0]
        nexus.results.cruise.torque           = Q[0][0]
        nexus.results.cruise.power            = P[0][0]
        nexus.results.cruise.power_c          = Cp[0][0]
        nexus.results.cruise.omega            = omega[0][0]
        nexus.results.cruise.thurst_c         = outputs.thrust_coefficient[0][0]
        nexus.results.cruise.max_sectional_cl = np.max(outputs.lift_coefficient[0]) 
        nexus.results.cruise.mean_CL          = np.mean(outputs.lift_coefficient[0])
        nexus.results.cruise.full_results     = outputs 
        nexus.results.cruise.efficiency       = etap[0][0]
        nexus.results.cruise.conditions       = conditions 
    else: 
        nexus.results.cruise.thrust           = 0.0
        nexus.results.cruise.torque           = 0.0
        nexus.results.cruise.power            = 0.0
        nexus.results.cruise.power_c          = 0.0
        nexus.results.cruise.thurst_c         = 0.0
        nexus.results.cruise.omega            = 0.0
        nexus.results.cruise.max_sectional_cl = 0.0
        nexus.results.cruise.mean_CL          = 0.0
        nexus.results.cruise.efficiency       = 0.0

    return nexus

# ----------------------------------------------------------------------
#   Run the hover noise
# ----------------------------------------------------------------------  
def run_hover_noise(nexus):
    
 
    rotors = nexus.vehicle_configurations.hover.networks.battery_propeller.propellers 
    rotor  = rotors.rotor  
    alpha  = rotor.optimization_parameters.multiobjective_aeroacoustic_weight
    
    if alpha != 1.0: 
        conditions   = nexus.results.hover.conditions 
        full_results = nexus.results.hover.full_results
    
        # microphone locations
        altitude            = rotor.hover.design_altitude
        ctrl_pts            = 1 
        theta               = rotor.optimization_parameters.noise_evaluation_angle 
        S_hover             = np.maximum(altitude,20*Units.feet)  
        mic_positions_hover = np.array([[0.0 , S_hover*np.sin(theta)  ,S_hover*np.cos(theta)]])      
        
        # Run noise model  
        conditions.noise.total_microphone_locations      = np.repeat(mic_positions_hover[ np.newaxis,:,: ],1,axis=0)
        conditions.aerodynamics.angle_of_attack          = np.ones((ctrl_pts,1))* 0. * Units.degrees 
        segment                                          = Segment() 
        segment.state.conditions                         = conditions
        segment.state.conditions.expand_rows(ctrl_pts)  
        noise                                            = SUAVE.Analyses.Noise.Fidelity_One() 
        settings                                         = noise.settings   
        num_mic                                          = len(conditions.noise.total_microphone_locations[0])  
        conditions.noise.number_of_microphones           = num_mic   
         
        propeller_noise_hover                        = propeller_mid_fidelity(rotors,full_results,segment,settings)   
        mean_SPL_hover                               = np.mean(propeller_noise_hover.SPL_dBA)     
            
        # Pack
        nexus.results.hover.mean_SPL   = mean_SPL_hover 
        nexus.results.hover.noise_data = propeller_noise_hover
    else:
        nexus.results.hover.mean_SPL  = 0
        nexus.results.hover.noise_data = None 

    return nexus


# ----------------------------------------------------------------------
#   Run the cruise noise
# ----------------------------------------------------------------------  
def run_cruise_noise(nexus):
    
    # unpack
    if nexus.prop_rotor: 
        
        rotors = nexus.vehicle_configurations.cruise.networks.battery_propeller.propellers 
        rotor  = rotors.rotor   
        alpha  = rotor.optimization_parameters.multiobjective_aeroacoustic_weight
        
        if alpha != 1.0:     
            conditions   = nexus.results.cruise.conditions 
            full_results = nexus.results.cruise.full_results
        
            # microphone locations
            altitude             = rotor.cruise.design_altitude
            ctrl_pts             = 1 
            theta                = rotor.optimization_parameters.noise_evaluation_angle 
            S_hover              = np.maximum(altitude,20*Units.feet)  
            mic_positions_cruise = np.array([[0.0 , S_hover*np.sin(theta)  ,S_hover*np.cos(theta)]])      
            
            # Run noise model  
            conditions.noise.total_microphone_locations      = np.repeat(mic_positions_cruise[ np.newaxis,:,: ],1,axis=0)
            conditions.aerodynamics.angle_of_attack          = np.ones((ctrl_pts,1))* 0. * Units.degrees 
            segment                                          = Segment() 
            segment.state.conditions                         = conditions
            segment.state.conditions.expand_rows(ctrl_pts)  
            noise                                            = SUAVE.Analyses.Noise.Fidelity_One() 
            settings                                         = noise.settings   
            num_mic                                          = len(conditions.noise.total_microphone_locations[0])  
            conditions.noise.number_of_microphones           = num_mic    
            
            propeller_noise_cruise                        = propeller_mid_fidelity(rotors,full_results,segment,settings)   
            mean_SPL_cruise                               = np.mean(propeller_noise_cruise.SPL_dBA)    
                
            # Pack
            nexus.results.cruise.mean_SPL   = mean_SPL_cruise 
            nexus.results.cruise.noise_data = propeller_noise_cruise
        else:
            nexus.results.cruise.mean_SPL   = 0 
            nexus.results.cruise.noise_data = None 
    else:
        nexus.results.cruise.mean_SPL   = 0 
        nexus.results.cruise.noise_data = None 
        
    return nexus

# ----------------------------------------------------------------------
#   Post Process Results to give back to the optimizer
# ----------------------------------------------------------------------   
def post_process(nexus):
     
    summary             = nexus.summary 
    rotor               = nexus.vehicle_configurations.hover.networks.battery_propeller.propellers.rotor  
    rotor_oei           = nexus.vehicle_configurations.oei.networks.battery_propeller.propellers.rotor  
    alpha               = rotor.optimization_parameters.multiobjective_aeroacoustic_weight
    beta                = rotor.optimization_parameters.multiobjective_performance_weight
    gamma               = rotor.optimization_parameters.multiobjective_acoustic_weight
    tol                 = rotor.optimization_parameters.tolerance 
    ideal_SPL           = rotor.optimization_parameters.ideal_SPL_dBA  
    ideal_efficiency    = rotor.optimization_parameters.ideal_efficiency      
    ideal_FoM           = rotor.optimization_parameters.ideal_figure_of_merit  
    print_iter          = nexus.print_iterations 
    
    summary.max_sectional_cl_hover  = nexus.results.hover.max_sectional_cl
    mean_CL_hover                   = nexus.results.hover.mean_CL
    omega_hover                     = nexus.results.hover.omega
    FM_hover                        = nexus.results.hover.figure_of_merit  
    
    # q to p ratios 
    summary.chord_p_to_q_ratio = rotor.chord_p/rotor.chord_q
    summary.twist_p_to_q_ratio = rotor.twist_p/rotor.twist_q    
        
    # blade taper consraint 
    c                                 = rotor.chord_distribution
    blade_taper                       = c[-1]/c[0]
    summary.blade_taper_constraint_1  = blade_taper  
    summary.blade_taper_constraint_2  = blade_taper  

    # blade twist constraint  
    beta_blade                      = rotor.twist_distribution 
    summary.blade_twist_constraint  = beta_blade[0] - beta_blade[-1]
    
    # OEI   
    if rotor.OEI.design_thrust == None: 
        summary.OEI_hover_thrust_power_residual =  tol*rotor.hover.design_thrust*1.1 -  abs(nexus.results.OEI.thrust - rotor.hover.design_thrust*1.1)  
    else:
        summary.OEI_hover_thrust_power_residual =  tol*rotor.OEI.design_thrust - abs(nexus.results.OEI.thrust - rotor.OEI.design_thrust)
            
    # thrust/power residual 
    if rotor.hover.design_thrust == None:
        summary.nominal_hover_thrust_power_residual = tol*rotor.hover.design_power - abs(nexus.results.hover.power - rotor.hover.design_power)
    else: 
        summary.nominal_hover_thrust_power_residual = tol*rotor.hover.design_thrust - abs(nexus.results.hover.thrust - rotor.hover.design_thrust)  
    
    if nexus.prop_rotor: 
        if rotor.cruise.design_thrust == None:
            summary.nominal_cruise_thrust_power_residual = tol*rotor.cruise.design_power  - abs(nexus.results.cruise.power - rotor.cruise.design_power) 
        else: 
            summary.nominal_cruise_thrust_power_residual = tol*rotor.cruise.design_thrust - abs(nexus.results.cruise.thrust - rotor.cruise.design_thrust)    
            
    # -------------------------------------------------------
    # OBJECTIVE FUNCTION
    # -------------------------------------------------------   
    performance_objective  = ((ideal_FoM - FM_hover)/ideal_FoM)*beta +  ((ideal_efficiency - nexus.results.cruise.efficiency)/ideal_efficiency)*(1-beta) 
    
    acoustic_objective     = ((nexus.results.hover.mean_SPL  - ideal_SPL)/ideal_SPL)*gamma  + ((nexus.results.cruise.mean_SPL - ideal_SPL)/ideal_SPL)*(1-gamma) 
 
    summary.objective      = performance_objective*alpha + acoustic_objective*(1-alpha)  
    

    if nexus.prop_rotor:  
        rotor_cru  = nexus.vehicle_configurations.cruise.networks.battery_propeller.propellers.rotor         
        summary.max_sectional_cl_cruise = nexus.results.cruise.max_sectional_cl   
        
    # -------------------------------------------------------
    # PRINT ITERATION PERFOMRMANCE
    # -------------------------------------------------------      
    if print_iter:
        print("Aeroacoustic Weight          : " + str(alpha))  
        print("Multiobj. Performance Weight : " + str(beta))  
        print("Multiobj. Acoustic Weight    : " + str(gamma)) 
        print("Performance Obj              : " + str(performance_objective))   
        print("Acoustic Obj                 : " + str(acoustic_objective))  
        print("Aeroacoustic Obj             : " + str(summary.objective))    
        print("Blade Taper                  : " + str(blade_taper))
        print("Hover RPM                    : " + str(omega_hover/Units.rpm))     
        if rotor.hover.design_thrust == None: 
            print("Hover Power                  : " + str(nexus.results.hover.power))  
        if rotor.hover.design_power == None: 
            print("Hover Thrust                 : " + str(nexus.results.hover.thrust))  
        print("Hover Average SPL            : " + str(nexus.results.hover.mean_SPL))    
        print("Hover Tip Mach               : " + str(rotor.hover.design_tip_mach)) 
        print("Hover Thrust/Power Residual  : " + str(summary.nominal_hover_thrust_power_residual)) 
        print("Hover Figure of Merit        : " + str(FM_hover))  
        print("Hover Max Sectional Cl       : " + str(summary.max_sectional_cl_hover)) 
        print("Hover Blade CL               : " + str(mean_CL_hover))    
        print("OEI Thrust                   : " + str(nexus.results.OEI.thrust)) 
        print("OEI Thrust/Power Residual    : " + str(summary.OEI_hover_thrust_power_residual)) 
        print("OEI Tip Mach                 : " + str(rotor_oei.OEI.design_tip_mach))  
        print("OEI Collective (deg)         : " + str(rotor_oei.inputs.pitch_command/Units.degrees)) 
        if nexus.prop_rotor:    
            print("Cruise RPM                   : " + str(nexus.results.cruise.omega/Units.rpm))    
            print("Cruise Collective (deg)      : " + str(rotor_cru.inputs.pitch_command/Units.degrees)) 
            if rotor_cru.cruise.design_thrust == None:  
                print("Cruise Power                 : " + str(nexus.results.cruise.power)) 
            if rotor_cru.cruise.design_power == None:  
                print("Cruise Thrust                : " + str(nexus.results.cruise.thrust))   
            print("Cruise Tip Mach              : " + str(rotor_cru.cruise.design_tip_mach))  
            print("Cruise Thrust/Power Residual : " + str(summary.nominal_cruise_thrust_power_residual))
            print("Cruise Efficiency            : " + str(nexus.results.cruise.efficiency)) 
            print("Cruise Max Sectional Cl      : " + str(summary.max_sectional_cl_cruise))  
            print("Cruise Blade CL              : " + str(nexus.results.cruise.mean_CL))  
        print("\n\n") 

   
    return nexus    
