## @ingroup Methods-Propulsion
# prop_rotor_design.py 
#
# Created: Feb 2022, M. Clarke

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  
# SUAVE Imports 
import SUAVE 
from SUAVE.Core                                                                           import Units, Data  
from SUAVE.Analyses.Mission.Segments.Segment                                              import Segment 
from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_mid_fidelity                    import propeller_mid_fidelity
import SUAVE.Optimization.Package_Setups.scipy_setup                                      as scipy_setup 
from SUAVE.Optimization                                                                   import Nexus      
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry  
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_polars  import compute_airfoil_polars 
from SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics                              import Aerodynamics 
from SUAVE.Analyses.Process                                                               import Process   

# Python package imports  
from numpy import linalg as LA  
import numpy as np 
import scipy as sp 
import time 

# ----------------------------------------------------------------------
#  Rotor Design
# ----------------------------------------------------------------------
## @ingroup Methods-Propulsion
def prop_rotor_design(prop_rotor,number_of_stations = 20, number_of_airfoil_section_points = 100,solver_name= 'SLSQP',
                      solver_sense_step = 1E-4,solver_tolerance = 1E-3):  
    """ Optimizes prop-rotor chord and twist given input parameters to meet either design power or thurst. 
        This scrip adopts SUAVE's native optimization style where the objective function is expressed 
        as an aeroacoustic function, considering both efficiency and radiated noise.
          
          Inputs: 
          prop_attributes.
              hub radius                       [m]
              tip radius                       [m]
              rotation rate                    [rad/s]
              freestream velocity              [m/s]
              number of blades                 [None]       
              number of stations               [None]
              design lift coefficient          [None]
              airfoil data                     [None]
              optimization_parameters.
                 slack_constaint               [None]
                 ideal_SPL_dbA                 [dBA]
                 aeroacoustic_weight           [None]
            
          Outputs:
          Twist distribution                   [array of radians]
          Chord distribution                   [array of meters]
              
          Assumptions: 
             N/A 
        
          Source:
             None 
    """    
    # Unpack prop-rotor geometry  
    N                     = number_of_stations       
    B                     = prop_rotor.number_of_blades    
    R                     = prop_rotor.tip_radius
    Rh                    = prop_rotor.hub_radius 
    design_thrust_cruise  = prop_rotor.design_thrust_cruise
    design_thrust_hover   = prop_rotor.design_thrust_hover
    design_power_cruise   = prop_rotor.design_power_cruise 
    design_power_hover    = prop_rotor.design_power_hover
    chi0                  = Rh/R  
    chi                   = np.linspace(chi0,1,N+1)  
    chi                   = chi[0:N]
    a_geo                 = prop_rotor.airfoil_geometry
    a_pol                 = prop_rotor.airfoil_polars        
    a_loc                 = prop_rotor.airfoil_polar_stations  
    
    # determine target values 
    if (design_thrust_hover == None) and (design_power_hover== None):
        raise AssertionError('Specify either design thrust or design power at hover!') 
    elif (design_thrust_hover!= None) and (design_power_hover!= None):
        raise AssertionError('Specify either design thrust or design power at hover!')  
    if (design_thrust_cruise == None) and (design_power_cruise== None):
        raise AssertionError('Specify either design thrust or design power at cruise!') 
    elif (design_thrust_cruise!= None) and (design_power_cruise!= None):
        raise AssertionError('Specify either design thrust or design power at cruise!')     
    if prop_rotor.rotation == None:
        prop_rotor.rotation = list(np.ones(int(B))) 
        
    # compute airfoil polars for airfoils 
    if  a_pol != None and a_loc != None:
        if len(a_loc) != N:
            raise AssertionError('\nDimension of airfoil sections must be equal to number of stations on prop-rotor')
        airfoil_polars  = compute_airfoil_polars(a_geo, a_pol)  
        cl_sur = airfoil_polars.lift_coefficient_surrogates 
        cd_sur = airfoil_polars.drag_coefficient_surrogates  
    else:
        raise AssertionError('\nDefine prop-rotor airfoil') 
  
    # append additional prop-rotor properties for optimization 
    airfoil_geometry_data                       = import_airfoil_geometry(prop_rotor.airfoil_geometry)  
    prop_rotor.number_of_blades                 = int(B)  
    prop_rotor.thickness_to_chord               = np.take(airfoil_geometry_data.thickness_to_chord,a_loc,axis=0)
    prop_rotor.radius_distribution              = chi*R
    prop_rotor.airfoil_cl_surrogates            = cl_sur
    prop_rotor.airfoil_cd_surrogates            = cd_sur 
    prop_rotor.airfoil_flag                     = True      
    prop_rotor.number_of_airfoil_section_points = number_of_airfoil_section_points
    
    # assign intial conditions for twist and chord distribution functions
    prop_rotor.chord_r  = 0.1*R     
    prop_rotor.chord_p  = 2         
    prop_rotor.chord_q  = 1         
    prop_rotor.chord_t  = 0.05*R    
    prop_rotor.twist_r  = np.pi/6   
    prop_rotor.twist_p  = 1         
    prop_rotor.twist_q  = 0.5       
    prop_rotor.twist_t  = 0   
    
    # start optimization   
    ti                   = time.time()     
    optimization_problem = rotor_optimization_setup(prop_rotor)  
    output               = scipy_setup.SciPy_Solve(optimization_problem,solver=solver_name, sense_step = solver_sense_step,tolerance = solver_tolerance)   
    Beta_c               = np.array([output[10],output[11]])    
    tf                   = time.time()
    elapsed_time         = round((tf-ti)/60,2)
    print('Prop-rotor Otimization Simulation Time: ' + str(elapsed_time) + ' mins')   
    
    # print optimization results 
    print (output)  
    
    # set remaining prop-rotor variables using optimized parameters 
    prop_rotor = set_optimized_rotor_planform(prop_rotor,optimization_problem,Beta_c)
    
    return prop_rotor
  
def rotor_optimization_setup(prop_rotor):
    """ Sets up prop-rotor optimization problem including design variables, constraints and objective function
        using SUAVE's Nexus optimization framework. Appends methodolody of planform modification to Nexus. 
          
          Inputs: 
              prop-rotor     - prop-rotor data structure  [None]
             
          Outputs: 
              nexus    - SUAVE's optimization framework   [None]
              
          Assumptions: 
            1) minimum allowable blade taper : 0.3  
            1) maximum allowable blade taper : 0.9     
        
          Source:
             None
    """    
    nexus                      = Nexus()
    problem                    = Data()
    nexus.optimization_problem = problem

    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------  
    R       = prop_rotor.tip_radius  
    tm_ll_h  = prop_rotor.design_tip_mach_range_hover[0]
    tm_ul_h  = prop_rotor.design_tip_mach_range_hover[1]    
    tm_0_h   = (tm_ul_h + tm_ll_h)/2 
    tm_ll_c  = prop_rotor.design_tip_mach_range_cruise[0]
    tm_ul_c  = prop_rotor.design_tip_mach_range_cruise[1]    
    tm_0_c   = (tm_ul_c + tm_ll_c)/2 
    PC_h     = prop_rotor.inputs.pitch_command_hover
    PC_cr    = prop_rotor.inputs.pitch_command_cruise  

    #  tag , initial, [lb,ub], scaling, units
    inputs    = [] 
    inputs.append([ 'chord_r'          ,  0.1*R    , 0.05*R     , 0.2*R     , 1.0     ,  1*Units.less])
    inputs.append([ 'chord_p'          ,  2        , 0.25       , 2.0       , 1.0     ,  1*Units.less])
    inputs.append([ 'chord_q'          ,  1        , 0.25       , 1.5       , 1.0     ,  1*Units.less])
    inputs.append([ 'chord_t'          ,  0.05*R   , 0.05*R     , 0.2*R     , 1.0     ,  1*Units.less])  
    inputs.append([ 'twist_r'          ,  np.pi/6  ,  0         , np.pi/4   , 1.0     ,  1*Units.less])
    inputs.append([ 'twist_p'          ,  1        , 0.25       , 2.0       , 1.0     ,  1*Units.less])
    inputs.append([ 'twist_q'          ,  0.5      , 0.25       , 1.5       , 1.0     ,  1*Units.less])
    inputs.append([ 'twist_t'          ,  np.pi/10 ,  0         , np.pi/4   , 1.0     ,  1*Units.less]) 
    inputs.append([ 'tip_mach_hover'   , tm_0_h    , tm_ll_h    , tm_ul_h   , 1.0     ,  1*Units.less])
    inputs.append([ 'tip_mach_cruise'  , tm_0_c    , tm_ll_c    , tm_ul_c   , 1.0     ,  1*Units.less]) 
    inputs.append([ 'pitch_cmd_hover'  , PC_h      , -np.pi/6   , np.pi/6   , 1.0     ,  1*Units.less])
    inputs.append([ 'pitch_cmd_cruise' , PC_cr     , -np.pi/6   , np.pi/6   , 1.0     ,  1*Units.less]) 
    problem.inputs = np.array(inputs,dtype=object)   

    # -------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------- 
    problem.objective = np.array([ 
                               # [ tag         , scaling, units ]
                                 [  'Aero_Acoustic_Obj'  ,  1.0   ,    1*Units.less] 
    ],dtype=object)
    
    # -------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------  
    constraints = [] 
    constraints.append([ 'thrust_power_residual_hover'  ,  '>'  ,  0.0 ,   1.0   , 1*Units.less])  
    constraints.append([ 'thrust_power_residual_cruise' ,  '>'  ,  0.0 ,   1.0   , 1*Units.less]) 
    constraints.append([ 'blade_taper_constraint_1'     ,  '>'  ,  0.3 ,   1.0   , 1*Units.less])  
    constraints.append([ 'blade_taper_constraint_2'     ,  '<'  ,  0.9 ,   1.0   , 1*Units.less])
    constraints.append([ 'blade_twist_constraint'       ,  '>'  ,  0.0 ,   1.0   , 1*Units.less])
    constraints.append([ 'max_sectional_cl_hover'       ,  '<'  ,  1.0 ,   1.0   , 1*Units.less])
    constraints.append([ 'max_sectional_cl_cruise'      ,  '<'  ,  0.8 ,   1.0   , 1*Units.less])
    constraints.append([ 'chord_p_to_q_ratio'           ,  '>'  ,  0.5 ,   1.0   , 1*Units.less])    
    constraints.append([ 'twist_p_to_q_ratio'           ,  '>'  ,  0.5 ,   1.0   , 1*Units.less])   
    problem.constraints =  np.array(constraints,dtype=object)                
    
    # -------------------------------------------------------------------
    #  Aliases
    # ------------------------------------------------------------------- 
    aliases = []
    aliases.append([ 'chord_r'                       , 'vehicle_configurations.*.networks.battery_propeller.propellers.prop_rotor.chord_r' ])
    aliases.append([ 'chord_p'                       , 'vehicle_configurations.*.networks.battery_propeller.propellers.prop_rotor.chord_p' ])
    aliases.append([ 'chord_q'                       , 'vehicle_configurations.*.networks.battery_propeller.propellers.prop_rotor.chord_q' ])
    aliases.append([ 'chord_t'                       , 'vehicle_configurations.*.networks.battery_propeller.propellers.prop_rotor.chord_t' ]) 
    aliases.append([ 'twist_r'                       , 'vehicle_configurations.*.networks.battery_propeller.propellers.prop_rotor.twist_r' ])
    aliases.append([ 'twist_p'                       , 'vehicle_configurations.*.networks.battery_propeller.propellers.prop_rotor.twist_p' ])
    aliases.append([ 'twist_q'                       , 'vehicle_configurations.*.networks.battery_propeller.propellers.prop_rotor.twist_q' ])
    aliases.append([ 'twist_t'                       , 'vehicle_configurations.*.networks.battery_propeller.propellers.prop_rotor.twist_t' ]) 
    aliases.append([ 'tip_mach_hover'                , 'vehicle_configurations.hover.networks.battery_propeller.propellers.prop_rotor.design_tip_mach_hover' ]) 
    aliases.append([ 'tip_mach_cruise'               , 'vehicle_configurations.cruise.networks.battery_propeller.propellers.prop_rotor.design_tip_mach_cruise' ]) 
    aliases.append([ 'pitch_cmd_hover'               , 'vehicle_configurations.hover.networks.battery_propeller.propellers.prop_rotor.inputs.pitch_command' ]) 
    aliases.append([ 'pitch_cmd_cruise'              , 'vehicle_configurations.cruise.networks.battery_propeller.propellers.prop_rotor.inputs.pitch_command' ]) 
    aliases.append([ 'Aero_Acoustic_Obj'             , 'summary.Aero_Acoustic_Obj'       ])  
    aliases.append([ 'thrust_power_residual_hover'   , 'summary.thrust_power_residual_hover'   ]) 
    aliases.append([ 'thrust_power_residual_cruise'  , 'summary.thrust_power_residual_cruise'   ]) 
    aliases.append([ 'blade_taper_constraint_1'      , 'summary.blade_taper_constraint_1'])  
    aliases.append([ 'blade_taper_constraint_2'      , 'summary.blade_taper_constraint_2'])   
    aliases.append([ 'max_sectional_cl_hover'        , 'summary.max_sectional_cl_hover'])    
    aliases.append([ 'max_sectional_cl_cruise'       , 'summary.max_sectional_cl_cruise'])  
    aliases.append([ 'blade_twist_constraint'        , 'summary.blade_twist_constraint'])   
    aliases.append([ 'chord_p_to_q_ratio'            , 'summary.chord_p_to_q_ratio'    ])  
    aliases.append([ 'twist_p_to_q_ratio'            , 'summary.twist_p_to_q_ratio'    ])     
    
    problem.aliases = aliases
    
    # -------------------------------------------------------------------
    #  Vehicles
    # -------------------------------------------------------------------
    nexus.vehicle_configurations = rotor_blade_setup(prop_rotor)
    
    # -------------------------------------------------------------------
    #  Analyses
    # -------------------------------------------------------------------
    nexus.analyses = None 
    
    # -------------------------------------------------------------------
    #  Missions
    # -------------------------------------------------------------------
    nexus.missions = None
    
    # -------------------------------------------------------------------
    #  Procedure
    # -------------------------------------------------------------------    
    nexus.procedure = optimization_procedure_set_up()
    
    # -------------------------------------------------------------------
    #  Summary
    # -------------------------------------------------------------------    
    nexus.summary = Data()     
    return nexus   
 
def set_optimized_rotor_planform(prop_rotor,optimization_problem,Beta_c):
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
    network_hover                   = optimization_problem.vehicle_configurations.hover.networks.battery_propeller
    network_cruise                  = optimization_problem.vehicle_configurations.cruise.networks.battery_propeller
    prop_rotor_opt_hover            = network_hover.propellers.prop_rotor 
    prop_rotor_opt_cruise           = network_cruise.propellers.prop_rotor 
    r                               = prop_rotor.radius_distribution
    R                               = prop_rotor.tip_radius     
    B                               = prop_rotor.number_of_blades  
    theta                           = prop_rotor_opt_hover.optimization_parameters.microphone_angle    
    prop_rotor.chord_distribution   = prop_rotor_opt_hover.chord_distribution
    prop_rotor.twist_distribution   = prop_rotor_opt_hover.twist_distribution
    c                               = prop_rotor.chord_distribution
    a_geo                           = prop_rotor.airfoil_geometry 
    a_loc                           = prop_rotor.airfoil_polar_stations  
    theta                           = prop_rotor.optimization_parameters.microphone_angle   

    # ------------------------------------------------------------------
    # PROP ROTOR HOVER ANALYSIS
    # ------------------------------------------------------------------  
    V_hover                         = prop_rotor.freestream_velocity_hover   
    alt_hover                       = prop_rotor.design_altitude_hover  
    atmosphere                      = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data_hover                 = atmosphere.compute_values(alt_hover)         
    omega_hover                     = prop_rotor_opt_hover.design_tip_mach_hover*atmo_data_hover.speed_of_sound[0]/prop_rotor.tip_radius   

    # microphone locations
    S_hover             = np.maximum(alt_hover,20*Units.feet)  
    mic_positions_hover = np.array([[0.0 , S_hover*np.sin(theta)  ,S_hover*np.cos(theta)]])  
                
    # Define run conditions 
    ctrl_pts                                         = 1   
    prop_rotor.inputs.omega                          = np.atleast_2d(omega_hover).T
    prop_rotor.inputs.pitch_command                  = Beta_c[0] 
    prop_rotor.orientation_euler_angles              = [0,np.pi/2,0]  
    conditions                                       = Aerodynamics()   
    conditions.freestream.update(atmo_data_hover)         
    conditions.frames.inertial.velocity_vector       = np.array([[0, 0. ,V_hover]]) 
    conditions.propulsion.throttle                   = np.ones((ctrl_pts,1))*1.0
    conditions.frames.body.transform_to_inertial     = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., -1.]]])   

    # Run rotor model 
    thrust_hover, _, power_hover, Cp_hover,noise_data_hover, _ = prop_rotor.spin(conditions)

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
    propeller_noise_hover                            = propeller_mid_fidelity(network_hover.propellers,noise_data_hover,segment,settings)   
    mean_SPL_hover                                   = np.mean(propeller_noise_hover.SPL_dBA) 
    
    
    # ------------------------------------------------------------------
    # PROP ROTOR CRUISE ANALYSIS
    # ------------------------------------------------------------------ 
    V_cruise                        = prop_rotor.freestream_velocity_cruise  
    alt_cruise                      = prop_rotor.design_altitude_cruise 
    atmosphere                      = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data_cruise                = atmosphere.compute_values(alt_cruise)    
    omega_cruise                    = prop_rotor_opt_cruise.design_tip_mach_cruise*atmo_data_cruise.speed_of_sound[0]/prop_rotor.tip_radius  

    # Microphone locations    
    S_cruise             = np.maximum(alt_cruise , 20*Units.feet) 
    mic_positions_cruise = np.array([[1E-3,S_cruise*np.sin(theta)  ,S_cruise*np.cos(theta)]])   

    # Set up for rotor model
    prop_rotor.inputs.omega                                = np.atleast_2d(omega_cruise).T
    prop_rotor.inputs.pitch_command                        = Beta_c[1]
    prop_rotor.orientation_euler_angles                    = [0,0,0]  
    conditions                                             = Aerodynamics()   
    conditions.freestream.update(atmo_data_cruise)         
    conditions.frames.inertial.velocity_vector             = np.array([[V_cruise, 0. ,0.]])
    conditions.propulsion.throttle                         = np.ones((ctrl_pts,1))*1.0
    conditions.frames.body.transform_to_inertial           = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])

    # Run rotor model 
    thrust_cruise , _, power_cruise, Cp_cruise  , noise_data_cruise , _ = prop_rotor.spin(conditions)

    # Run noise Model 
    conditions.noise.total_microphone_locations            = np.repeat(mic_positions_cruise[ np.newaxis,:,: ],1,axis=0)
    conditions.aerodynamics.angle_of_attack                = np.ones((ctrl_pts,1))* 0. * Units.degrees 
    segment                                                = Segment() 
    segment.state.conditions                               = conditions
    segment.state.conditions.expand_rows(ctrl_pts)  
    noise                                                  = SUAVE.Analyses.Noise.Fidelity_One() 
    settings                                               = noise.settings   
    num_mic                                                = len(conditions.noise.total_microphone_locations[0])  
    conditions.noise.number_of_microphones                 = num_mic    
    propeller_noise_cruise                                 = propeller_mid_fidelity(network_cruise.propellers,noise_data_cruise,segment,settings)   
    mean_SPL_cruise                                        = np.mean(propeller_noise_cruise.SPL_dBA) 

    if prop_rotor.design_power_hover   == None: 
        prop_rotor.design_power_hover  = power_hover[0][0]
    if prop_rotor.design_thrust_hover  == None: 
        prop_rotor.design_thrust_hover = -thrust_hover[0][2] 
        
    if prop_rotor.design_power_cruise   == None: 
        prop_rotor.design_power_cruise  = power_cruise[0][0]
    if prop_rotor.design_thrust_cruise  == None: 
        prop_rotor.design_thrust_cruise = thrust_cruise[0][0]
                  
    blade_area            = sp.integrate.cumtrapz(B*c, r-r[0])
    sigma                 = blade_area[-1]/(np.pi*R**2)    
    MCA                   = c/4. - c[0]/4.
    airfoil_geometry_data = import_airfoil_geometry(a_geo) 
    t_max                 = np.take(airfoil_geometry_data.max_thickness,a_loc,axis=0)*c 
    t_c                   =  np.take(airfoil_geometry_data.thickness_to_chord,a_loc,axis=0)  
    
    prop_rotor.design_torque_hover               = power_hover[0][0]/omega_hover
    prop_rotor.design_torque_cruise              = power_cruise[0][0]/omega_cruise 
    prop_rotor.design_torque                     = power_hover[0][0]/omega_hover  
    prop_rotor.max_thickness_distribution        = t_max 
    prop_rotor.radius_distribution               = r 
    prop_rotor.number_of_blades                  = int(B) 
    prop_rotor.design_power_coefficient_hover    = Cp_hover[0][0] 
    prop_rotor.design_power_coefficient_cruise   = Cp_cruise[0][0] 
    prop_rotor.design_thrust_coefficient_hover   = noise_data_hover.thrust_coefficient[0][0] 
    prop_rotor.design_thrust_coefficient_cruise  = noise_data_cruise.thrust_coefficient[0][0] 
    prop_rotor.mid_chord_alignment               = MCA
    prop_rotor.thickness_to_chord                = t_c 
    prop_rotor.design_SPL_dBA_hover              = mean_SPL_hover
    prop_rotor.design_SPL_dBA_cruise             = mean_SPL_cruise
    prop_rotor.design_performance_hover          = noise_data_hover
    prop_rotor.design_performance_cruise         = noise_data_cruise
    prop_rotor.design_acoustics_hover            = propeller_noise_hover
    prop_rotor.design_acoustics_cruise           = propeller_noise_cruise
    prop_rotor.blade_solidity                    = sigma   
    prop_rotor.inputs.pitch_command_hover        = Beta_c[0]      
    prop_rotor.inputs.pitch_command_cruise       = Beta_c[1]
    prop_rotor.angular_velocity_hover            = omega_hover       
    prop_rotor.angular_velocity_cruise           = omega_cruise 
    prop_rotor.angular_velocity                  = omega_hover  
    prop_rotor.airfoil_flag                      = True    
    
    return prop_rotor 

def rotor_blade_setup(prop_rotor): 
    """ Defines a dummy vehicle for prop-rotor blade optimization.
          
          Inputs:  
             prop-rotor   - prop-rotor data structure             [None] 
              
          Outputs:  
             configs      - configuration used in optimization    [None]
              
          Assumptions: 
             N/A 
        
          Source:
             None
    """    
    vehicle                            = SUAVE.Vehicle()  
    net                                = SUAVE.Components.Energy.Networks.Battery_Propeller()
    net.number_of_propeller_engines    = 1
    net.identical_propellers           = True  
    net.propellers.append(prop_rotor)  
    vehicle.append_component(net)
    
    configs                             = SUAVE.Components.Configs.Config.Container()
    base_config                         = SUAVE.Components.Configs.Config(vehicle)
    
    config                              = SUAVE.Components.Configs.Config(base_config)
    config.tag                          = 'cruise'
    configs.append(config)
    
    config                              = SUAVE.Components.Configs.Config(base_config)
    config.tag                          = 'hover' 
    config.networks.battery_propeller.propellers.prop_rotor.orientation_euler_angles = [0,np.pi/2,0]    
    configs.append(config)       
    return configs   
     
def optimization_procedure_set_up():
    """ Defines the procedure for planform modifications and  aeroacoustic analyses.
          
          Inputs:  
             N/A
              
          Outputs:  
             procedure - optimization methodology [None]
              
          Assumptions: 
             N/A 
        
          Source:
             None
    """    
    procedure                   = Process()
    procedure.modify_rotor      = modify_blade_geometry
    procedure.post_process      = post_process    
    return procedure  
 
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
    vehicle_cruise    = nexus.vehicle_configurations.cruise
    prop_rotor_hover  = vehicle_hover.networks.battery_propeller.propellers.prop_rotor 
    prop_rotor_cruise = vehicle_cruise.networks.battery_propeller.propellers.prop_rotor 
    
    # Update geometry of blade
    c       = updated_blade_geometry(prop_rotor_hover.radius_distribution/prop_rotor_hover.tip_radius ,prop_rotor_hover.chord_r,prop_rotor_hover.chord_p,prop_rotor_hover.chord_q,prop_rotor_hover.chord_t)     
    beta    = updated_blade_geometry(prop_rotor_hover.radius_distribution/prop_rotor_hover.tip_radius ,prop_rotor_hover.twist_r,prop_rotor_hover.twist_p,prop_rotor_hover.twist_q,prop_rotor_hover.twist_t)   
    
    prop_rotor_hover.chord_distribution          = c
    prop_rotor_cruise.chord_distribution         = prop_rotor_hover.chord_distribution
    prop_rotor_hover.twist_distribution          = beta  
    prop_rotor_cruise.twist_distribution         = prop_rotor_hover.twist_distribution
    prop_rotor_hover.mid_chord_alignment         = c/4. - c[0]/4.
    prop_rotor_cruise.mid_chord_alignment        = prop_rotor_hover.mid_chord_alignment
    airfoil_geometry_data                        = import_airfoil_geometry(prop_rotor_hover.airfoil_geometry) 
    t_max                                        = np.take(airfoil_geometry_data.max_thickness,prop_rotor_hover.airfoil_polar_stations,axis=0)*c
    prop_rotor_hover.airfoil_data                = airfoil_geometry_data
    prop_rotor_cruise.airfoil_data               = prop_rotor_hover.airfoil_data
    prop_rotor_hover.max_thickness_distribution  = t_max   
    prop_rotor_cruise.max_thickness_distribution = prop_rotor_hover.max_thickness_distribution
     
    # diff the new data
    vehicle_hover.store_diff() 
    vehicle_cruise.store_diff() 
    
    return nexus    

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
    b       = chi[-1]
    r       = len(chi)                
    n       = np.linspace(r-1,0,r)          
    theta_n = n*(np.pi/2)/r              
    y_n     = b*np.cos(theta_n)          
    eta_n   = np.abs(y_n/b)            
    x_cos   = c_r*(1 - eta_n**p)**q + c_t*eta_n 
    x_lin   = np.interp(chi,eta_n, x_cos)  
    return x_lin 

def post_process(nexus):
    """ Performs aerodynamic and aeroacoustic analysis on prop-rotor blade, computes constraint
        violations and objective function.
          
          Inputs:  
             nexus     - SUAVE optmization framework with prop-rotor blade data structure [None]
              
          Outputs:  
             nexus     - SUAVE optmization framework with prop-rotor blade data structure [None]
              
          Assumptions: 
             N/A 
        
          Source:
             N/A
    """    
    summary              = nexus.summary  
    
    # -------------------------------------------------------
    # RUN AEROACOUSTICS MODELS IN HOVER 
    # -------------------------------------------------------    
    # unpack prop-rotor properties 
    vehicle_hover        = nexus.vehicle_configurations.hover  
    propellers_hover     = vehicle_hover.networks.battery_propeller.propellers
    prop_rotor_hover     = propellers_hover.prop_rotor  
    beta_blade           = prop_rotor_hover.twist_distribution 
    c                    = prop_rotor_hover.chord_distribution  
    theta                = prop_rotor_hover.optimization_parameters.microphone_angle
    alpha                = prop_rotor_hover.optimization_parameters.aeroacoustic_weight
    beta                 = prop_rotor_hover.optimization_parameters.multiobjective_performance_weight
    gamma                = prop_rotor_hover.optimization_parameters.multiobjective_acoustic_weight
    epsilon              = prop_rotor_hover.optimization_parameters.slack_constaint 
    ideal_SPL            = prop_rotor_hover.optimization_parameters.ideal_SPL_dBA  
    atmosphere           = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    
    # Calculate atmospheric properties
    V_hover              = prop_rotor_hover.freestream_velocity_hover   
    alt_hover            = prop_rotor_hover.design_altitude_hover
    atmo_data            = atmosphere.compute_values(alt_hover)   
    omega_hover          = prop_rotor_hover.design_tip_mach_hover*atmo_data.speed_of_sound[0]/prop_rotor_hover.tip_radius   
 
    # microphone locations
    S_hover             = np.maximum(alt_hover,20*Units.feet)  
    mic_positions_hover = np.array([[0.0 , S_hover*np.sin(theta)  ,S_hover*np.cos(theta)]])  
                
    # Define run conditions 
    ctrl_pts                                         = 1   
    prop_rotor_hover.inputs.omega                    = np.atleast_2d(omega_hover).T
    conditions                                       = Aerodynamics()   
    conditions.freestream.update(atmo_data)         
    conditions.frames.inertial.velocity_vector       = np.array([[0, 0. ,V_hover]]) 
    conditions.propulsion.throttle                   = np.ones((ctrl_pts,1))*1.0
    conditions.frames.body.transform_to_inertial     = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., -1.]]])  

    # Run rotor model 
    thrust_hover,_, power_hover, Cp_hover,noise_data_hover, _ = prop_rotor_hover.spin(conditions) 

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
    
    # Run noise model if necessary    
    if alpha != 1.0: 
        try: 
            propeller_noise       = propeller_mid_fidelity(propellers_hover,noise_data_hover,segment,settings)   
            Acoustic_Metric_hover = np.mean(propeller_noise.SPL_dBA)  
        except:
            Acoustic_Metric_hover = 100 
    else:
        Acoustic_Metric_hover  = 0  
    
    # -------------------------------------------------------
    # RUN AEROACOUSTICS MODELS IN CRUISE
    # -------------------------------------------------------    
    # unpack prop-rotor properties 
    vehicle_cruise       = nexus.vehicle_configurations.cruise
    propellers_cruise    = vehicle_cruise.networks.battery_propeller.propellers
    prop_rotor_cruise    = propellers_cruise.prop_rotor     
    V_cruise             = prop_rotor_cruise.freestream_velocity_cruise   
    alt_cruise           = prop_rotor_cruise.design_altitude_cruise  
    atmo_data            = atmosphere.compute_values(alt_cruise)     
    omega_cruise         = prop_rotor_cruise.design_tip_mach_cruise*atmo_data.speed_of_sound[0]/prop_rotor_cruise.tip_radius    
    
    # Microhpone locations  
    S_cruise             = np.maximum(alt_cruise, 20*Units.feet) 
    mic_positions_cruise = np.array([[1E-3,S_cruise*np.sin(theta)  ,S_cruise*np.cos(theta)]])   

    # Define run conditions 
    prop_rotor_cruise.inputs.omega                   = np.atleast_2d(omega_cruise).T
    conditions                                       = Aerodynamics()   
    conditions.freestream.update(atmo_data)         
    conditions.frames.inertial.velocity_vector       = np.array([[V_cruise, 0. ,0.]])
    conditions.propulsion.throttle                   = np.ones((ctrl_pts,1))*1.0
    conditions.frames.body.transform_to_inertial     = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])

    # Run Propeller model 
    thrust_cruise ,_, power_cruise, Cp_cruise  , noise_data_cruise , etap_cruise  = prop_rotor_cruise.spin(conditions)  

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
    
    # Run noise model if necessary 
    if alpha != 1.0: 
        if gamma != 1: 
            try: 
                propeller_noise_cruise = propeller_mid_fidelity(propellers_cruise,noise_data_cruise,segment,settings)  
                Acoustic_Metric_cruise = np.mean(propeller_noise_cruise.SPL_dBA) 
            except:
                Acoustic_Metric_cruise = 100 
    else:
        Acoustic_Metric_cruise = 0    
        
    # -------------------------------------------------------
    # CONTRAINTS
    # -------------------------------------------------------
    # thrust/power constraints
    if prop_rotor_hover.design_thrust_hover == None:
        summary.thrust_power_residual_hover = epsilon*prop_rotor_hover.design_power_hover \
            - abs(power_hover[0][0] - prop_rotor_hover.design_power_hover)
    else: 
        summary.thrust_power_residual_hover = epsilon*prop_rotor_hover.design_thrust_hover \
            - abs(-thrust_hover[0][2] - prop_rotor_hover.design_thrust_hover) 
    if prop_rotor_hover.design_thrust_cruise == None:
        summary.thrust_power_residual_cruise = epsilon*prop_rotor_cruise.design_power_cruise \
            - abs(power_cruise[0][0] - prop_rotor_cruise.design_power_cruise) 
    else: 
        summary.thrust_power_residual_cruise = epsilon*prop_rotor_cruise.design_thrust_cruise \
            - abs(thrust_cruise[0][0] - prop_rotor_cruise.design_thrust_cruise) 
        
    # q to p ratios 
    summary.chord_p_to_q_ratio = prop_rotor_cruise.chord_p/prop_rotor_cruise.chord_q
    summary.twist_p_to_q_ratio = prop_rotor_cruise.twist_p/prop_rotor_cruise.twist_q
    
    # Cl constraint  
    summary.max_sectional_cl_hover    = np.max(noise_data_hover.lift_coefficient[0])
    summary.max_sectional_cl_cruise   = np.max(noise_data_cruise.lift_coefficient[0])
    mean_CL_hover                     = np.mean(noise_data_hover.lift_coefficient[0])
    mean_CL_cruise                    = np.mean(noise_data_cruise.lift_coefficient[0])
    
    # blade taper consraint 
    blade_taper                       = c[-1]/c[0]
    summary.blade_taper_constraint_1  = blade_taper 
    summary.blade_taper_constraint_2  = blade_taper
    
    # figure of merit for hover 
    C_t_UIUC        = noise_data_hover.thrust_coefficient[0][0]
    C_t_rot         = C_t_UIUC*8/(np.pi**3)
    C_p_UIUC        = Cp_hover[0][0] 
    C_q_UIUC        = C_p_UIUC/(2*np.pi) 
    C_q_rot         = C_q_UIUC*16/(np.pi**3)   
    C_p_rot         = C_q_rot 
    ideal_FM_hover  = 1
    FM_hover        = ((C_t_rot**1.5)/np.sqrt(2))/C_p_rot 
    
    # efficiency for cruise 
    ideal_eta_cruise = 1 
    eta_cruise       = etap_cruise[0][0] 

    # blade twist consraint  
    summary.blade_twist_constraint = beta_blade[0] - beta_blade[-1] 
    
    # -------------------------------------------------------
    # OBJECTIVE FUNCTION
    # ------------------------------------------------------- 
    aero_objective  = LA.norm((FM_hover - ideal_FM_hover)*100/(ideal_FM_hover*100))*beta \
        +  LA.norm((eta_cruise - ideal_eta_cruise)*100/(ideal_eta_cruise*100))*(1-beta) 
    
    acous_objective = LA.norm((Acoustic_Metric_hover - ideal_SPL)/ideal_SPL)*gamma + \
         LA.norm((Acoustic_Metric_cruise - ideal_SPL)/ideal_SPL)*(1-gamma)
        
    summary.Aero_Acoustic_Obj =  aero_objective*10*alpha + acous_objective*10*(1-alpha)
        
    # -------------------------------------------------------
    # PRINT ITERATION PERFOMRMANCE
    # -------------------------------------------------------                
    print("Aeroacoustic Obj             : " + str(summary.Aero_Acoustic_Obj))     
    print("Aeroacoustic Weight          : " + str(alpha))  
    print("Multiobj. Performance Weight : " + str(beta))  
    print("Multiobj. Acoustic Weight    : " + str(gamma)) 
    print("Blade Taper                  : " + str(blade_taper))
    print("Hover RPM                    : " + str(omega_hover[0]/Units.rpm))    
    print("Hover Pitch Command (deg)    : " + str(prop_rotor_hover.inputs.pitch_command/Units.degrees)) 
    if prop_rotor_hover.design_thrust == None: 
        print("Hover Power                  : " + str(power_hover[0][0]))  
    if prop_rotor_hover.design_power == None: 
        print("Hover Thrust                 : " + str(-thrust_hover[0][2]))  
    print("Hover Average SPL            : " + str(Acoustic_Metric_hover))    
    print("Hover Tip Mach               : " + str(prop_rotor_hover.design_tip_mach_hover))  
    print("Hover Thrust/Power Residual  : " + str(summary.thrust_power_residual_hover)) 
    print("Hover Figure of Merit        : " + str(FM_hover))  
    print("Hover Max Sectional Cl       : " + str(summary.max_sectional_cl_hover)) 
    print("Hover Blade CL               : " + str(mean_CL_hover))   
    print("Cruise RPM                   : " + str(omega_cruise[0]/Units.rpm))    
    print("Cruise Pitch Command (deg)   : " + str(prop_rotor_cruise.inputs.pitch_command/Units.degrees)) 
    if prop_rotor_cruise.design_thrust == None:  
        print("Cruise Power                 : " + str(power_cruise[0][0])) 
    if prop_rotor_cruise.design_power == None:  
        print("Cruise Thrust                : " + str(thrust_cruise[0][0]))   
    print("Cruise Tip Mach              : " + str(prop_rotor_cruise.design_tip_mach_cruise))  
    print("Cruise Thrust/Power Residual : " + str(summary.thrust_power_residual_cruise))
    print("Cruise Efficiency            : " + str(eta_cruise)) 
    print("Cruise Max Sectional Cl      : " + str(summary.max_sectional_cl_cruise))  
    print("Cruise Blade CL              : " + str(mean_CL_cruise))  
    print("\n\n") 
    
    return nexus 