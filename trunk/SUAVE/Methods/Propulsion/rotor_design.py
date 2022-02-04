## @ingroup Methods-Propulsion
# rotor_design.py 
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
def rotor_design(rotor,number_of_stations = 20, number_of_airfoil_section_points = 100,solver_name= 'SLSQP'):  
    """ Optimizes rotor chord and twist given input parameters to meet either design power or thurst. 
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
          Twist distribution                 [array of radians]
          Chord distribution                 [array of meters]
              
          Assumptions: 
             N/A 
        
          Source:
             None 
    """    
    # Unpack rotor geometry  
    N             = number_of_stations       
    B             = rotor.number_of_blades    
    R             = rotor.tip_radius
    Rh            = rotor.hub_radius 
    design_thrust = rotor.design_thrust
    design_power  = rotor.design_power
    chi0          = Rh/R  
    chi           = np.linspace(chi0,1,N+1)  
    chi           = chi[0:N]
    a_geo         = rotor.airfoil_geometry
    a_pol         = rotor.airfoil_polars        
    a_loc         = rotor.airfoil_polar_stations  
    
    # determine target values 
    if (design_thrust == None) and (design_power== None):
        raise AssertionError('Specify either design thrust or design power!') 
    elif (design_thrust!= None) and (design_power!= None):
        raise AssertionError('Specify either design thrust or design power!') 
    if rotor.rotation == None:
        rotor.rotation = list(np.ones(int(B))) 
        
    # compute airfoil polars for airfoils 
    if  a_pol != None and a_loc != None:
        if len(a_loc) != N:
            raise AssertionError('\nDimension of airfoil sections must be equal to number of stations on rotor')
        airfoil_polars  = compute_airfoil_polars(a_geo, a_pol)  
        cl_sur = airfoil_polars.lift_coefficient_surrogates 
        cd_sur = airfoil_polars.drag_coefficient_surrogates  
    else:
        raise AssertionError('\nDefine rotor airfoil') 
  
    # append additional rotor properties for optimization 
    airfoil_geometry_data                  = import_airfoil_geometry(rotor.airfoil_geometry)  
    rotor.number_of_blades                 = int(B)  
    rotor.thickness_to_chord               = np.take(airfoil_geometry_data.thickness_to_chord,a_loc,axis=0)
    rotor.radius_distribution              = chi*R
    rotor.airfoil_cl_surrogates            = cl_sur
    rotor.airfoil_cd_surrogates            = cd_sur 
    rotor.airfoil_flag                     = True      
    rotor.number_of_airfoil_section_points = number_of_airfoil_section_points
    
    # assign intial conditions for twist and chord distribution functions
    rotor.chord_r                          = 0.1*R     
    rotor.chord_p                          = 1.0       
    rotor.chord_q                          = 0.5       
    rotor.chord_t                          = 0.05*R    
    rotor.twist_r                          = np.pi/6   
    rotor.twist_p                          = 1.0       
    rotor.twist_q                          = 0.5       
    rotor.twist_t                          = np.pi/10   
    
    # start optimization 
    ti = time.time()   
    optimization_problem = rotor_optimization_setup(rotor) 
    output = scipy_setup.SciPy_Solve(optimization_problem,solver=solver_name, sense_step = 1E-4, tolerance = 1E-3)    
    tf           = time.time()
    elapsed_time = round((tf-ti)/60,2)
    print('Rotor Otimization Simulation Time: ' + str(elapsed_time))   
    
    # print optimization results 
    print (output)  
    
    # set remaining rotor variables using optimized parameters 
    rotor = set_optimized_rotor_planform(rotor,optimization_problem)
    
    return rotor
  
def rotor_optimization_setup(rotor):
    """ Sets up rotor optimization problem including design variables, constraints and objective function
        using SUAVE's Nexus optimization framework. Appends methodolody of planform modification to Nexus. 
          
          Inputs: 
             rotor     - rotor data structure           [None]
             
          Outputs: 
              nexus    - SUAVE's optimization framework [None]
              
          Assumptions: 
            1) minimum allowable blade taper : 0.2  
            1) maximum allowable blade taper : 0.7     
        
          Source:
             None
    """    
    nexus                      = Nexus()
    problem                    = Data()
    nexus.optimization_problem = problem

    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------  
    R      = rotor.tip_radius  
    inputs = []
    inputs.append([ 'chord_r'    , 0.1*R     , 0.05*R , 0.2*R    , 1.0     ,  1*Units.less])
    inputs.append([ 'chord_p'    , 2         , 0.25   , 2.0      , 1.0     ,  1*Units.less])
    inputs.append([ 'chord_q'    , 1         , 0.25   , 1.5      , 1.0     ,  1*Units.less])
    inputs.append([ 'chord_t'    , 0.05*R    , 0.05*R , 0.2*R    , 1.0     ,  1*Units.less])  
    inputs.append([ 'twist_r'    , np.pi/6   ,  0     , np.pi/4  , 1.0     ,  1*Units.less])
    inputs.append([ 'twist_p'    , 1         , 0.25   , 2.0      , 1.0     ,  1*Units.less])
    inputs.append([ 'twist_q'    , 0.5       , 0.25   , 1.5      , 1.0     ,  1*Units.less])
    inputs.append([ 'twist_t'    , np.pi/10  ,  0     , np.pi/4  , 1.0     ,  1*Units.less]) 
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
    constraints.append([ 'thrust_power_residual'    ,  '>'  ,  0.0 ,   1.0   , 1*Units.less]), 
    constraints.append([ 'blade_taper_constraint_1' ,  '>'  ,  0.2 ,   1.0   , 1*Units.less]), 
    constraints.append([ 'blade_taper_constraint_2' ,  '<'  ,  0.7 ,   1.0   , 1*Units.less]),  
    constraints.append([ 'chord_p_to_q_ratio'       ,  '>'  ,  0.5 ,   1.0   , 1*Units.less])   
    constraints.append([ 'twist_p_to_q_ratio'       ,  '>'  ,  0.5 ,   1.0   , 1*Units.less])   
    problem.constraints =  np.array(constraints,dtype=object)                
    
    # -------------------------------------------------------------------
    #  Aliases
    # ------------------------------------------------------------------- 
    aliases = []
    aliases.append([ 'chord_r'                   , 'vehicle_configurations.*.networks.battery_propeller.lift_rotors.rotor.chord_r' ])
    aliases.append([ 'chord_p'                   , 'vehicle_configurations.*.networks.battery_propeller.lift_rotors.rotor.chord_p' ])
    aliases.append([ 'chord_q'                   , 'vehicle_configurations.*.networks.battery_propeller.lift_rotors.rotor.chord_q' ])
    aliases.append([ 'chord_t'                   , 'vehicle_configurations.*.networks.battery_propeller.lift_rotors.rotor.chord_t' ]) 
    aliases.append([ 'twist_r'                   , 'vehicle_configurations.*.networks.battery_propeller.lift_rotors.rotor.twist_r' ])
    aliases.append([ 'twist_p'                   , 'vehicle_configurations.*.networks.battery_propeller.lift_rotors.rotor.twist_p' ])
    aliases.append([ 'twist_q'                   , 'vehicle_configurations.*.networks.battery_propeller.lift_rotors.rotor.twist_q' ])
    aliases.append([ 'twist_t'                   , 'vehicle_configurations.*.networks.battery_propeller.lift_rotors.rotor.twist_t' ]) 
    aliases.append([ 'Aero_Acoustic_Obj'         , 'summary.Aero_Acoustic_Obj'       ])  
    aliases.append([ 'thrust_power_residual'     , 'summary.thrust_power_residual'   ]) 
    aliases.append([ 'blade_taper_constraint_1'  , 'summary.blade_taper_constraint_1'    ])  
    aliases.append([ 'blade_taper_constraint_2'  , 'summary.blade_taper_constraint_2'    ])  
    aliases.append([ 'chord_p_to_q_ratio'        , 'summary.chord_p_to_q_ratio'    ])  
    aliases.append([ 'twist_p_to_q_ratio'        , 'summary.twist_p_to_q_ratio'    ])     
    
    problem.aliases = aliases
    
    # -------------------------------------------------------------------
    #  Vehicles
    # -------------------------------------------------------------------
    nexus.vehicle_configurations = rotor_blade_setup(rotor)
    
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
 
def set_optimized_rotor_planform(rotor,optimization_problem):
    """ Append parameters of optimized rotor to input rotor
          
          Inputs:  
             rotor                - rotor data structure                   [None]
             optimization_problem - data struction of optimized parameters [None]
             
          Outputs: 
             rotor                - rotor data structure                   [None]
              
          Assumptions: 
             1) Noise measurements are taken at 90, 112.5 and 135 degrees from rotor plane
             2) Distances from microphone and rotor are all 10m
        
          Source:
             None
    """    
    r                        = rotor.radius_distribution
    R                        = rotor.tip_radius     
    chi                      = r/R 
    B                        = rotor.number_of_blades 
    omega                    = rotor.angular_velocity
    V                        = rotor.freestream_velocity  
    alt                      = rotor.design_altitude
    network                  = optimization_problem.vehicle_configurations.rotor_testbench.networks.battery_propeller
    rotor_opt                = network.lift_rotors.rotor 
    rotor.chord_distribution = rotor_opt.chord_distribution
    rotor.twist_distribution = rotor_opt.twist_distribution
    c                        = rotor.chord_distribution
    a_geo                    = rotor.airfoil_geometry 
    a_loc                    = rotor.airfoil_polar_stations  
  
    # Calculate atmospheric properties
    atmosphere     = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data      = atmosphere.compute_values(alt)   
    T              = atmo_data.temperature[0]
    rho            = atmo_data.density[0]
    a              = atmo_data.speed_of_sound[0]
    mu             = atmo_data.dynamic_viscosity[0]  
    ctrl_pts       = 1 

    # Run Conditions     
    theta  = np.array([90,112.5,135])*Units.degrees + 1E-1
    S      = 10.  

    # microphone locations
    positions = np.zeros(( len(theta),3))
    for i in range(len(theta)):
        if theta[i]*Units.degrees < np.pi/2:
            positions[i][:] = [-S*np.cos(theta[i]*Units.degrees)  ,S*np.sin(theta[i]*Units.degrees), 0.0]
        else: 
            positions[i][:] = [S*np.sin(theta[i]*Units.degrees- np.pi/2)  ,S*np.cos(theta[i]*Units.degrees - np.pi/2), 0.0] 
            
    # Set up for Propeller Model
    rotor.inputs.omega                                     = np.atleast_2d(omega).T
    conditions                                             = Aerodynamics()   
    conditions.freestream.density                          = np.ones((ctrl_pts,1)) * rho
    conditions.freestream.dynamic_viscosity                = np.ones((ctrl_pts,1)) * mu
    conditions.freestream.speed_of_sound                   = np.ones((ctrl_pts,1)) * a 
    conditions.freestream.temperature                      = np.ones((ctrl_pts,1)) * T 
    conditions.frames.inertial.velocity_vector             = np.array([[V, 0. ,0.]])
    conditions.propulsion.throttle                         = np.ones((ctrl_pts,1))*1.0
    conditions.frames.body.transform_to_inertial           = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])

    # Run Propeller model 
    thrust , torque, power, Cp  , noise_data , etap        = rotor.spin(conditions)

    # Prepare Inputs for Noise Model  
    conditions.noise.total_microphone_locations            = np.repeat(positions[ np.newaxis,:,: ],1,axis=0)
    conditions.aerodynamics.angle_of_attack                = np.ones((ctrl_pts,1))* 0. * Units.degrees 
    segment                                                = Segment() 
    segment.state.conditions                               = conditions
    segment.state.conditions.expand_rows(ctrl_pts) 

    # Store Noise Data 
    noise                                                  = SUAVE.Analyses.Noise.Fidelity_One() 
    settings                                               = noise.settings   
    num_mic                                                = len(conditions.noise.total_microphone_locations[0])  
    conditions.noise.number_of_microphones                 = num_mic   

    propeller_noise   = propeller_mid_fidelity(network.lift_rotors,noise_data,segment,settings)   
    mean_SPL          =  np.mean(propeller_noise.SPL_dBA) 
    
    if rotor.design_power == None: 
        rotor.design_power = power[0][0]
    if rotor.design_thrust == None: 
        rotor.design_thrust = thrust[0][0]
        
    design_torque = power[0][0]/omega
    
    # blade solidity
    r          = chi*R                    # Radial coordinate   
    blade_area = sp.integrate.cumtrapz(B*c, r-r[0])
    sigma      = blade_area[-1]/(np.pi*R**2)   
    
    MCA    = c/4. - c[0]/4.
    airfoil_geometry_data = import_airfoil_geometry(a_geo) 
    t_max = np.take(airfoil_geometry_data.max_thickness,a_loc,axis=0)*c 
    t_c   =  np.take(airfoil_geometry_data.thickness_to_chord,a_loc,axis=0)  
    
    rotor.design_torque              = design_torque
    rotor.max_thickness_distribution = t_max 
    rotor.radius_distribution        = r 
    rotor.number_of_blades           = int(B) 
    rotor.design_power_coefficient   = Cp[0][0] 
    rotor.design_thrust_coefficient  = noise_data.thrust_coefficient[0][0] 
    rotor.mid_chord_alignment        = MCA
    rotor.thickness_to_chord         = t_c 
    rotor.design_SPL_dBA             = mean_SPL
    rotor.blade_solidity             = sigma   
    rotor.airfoil_flag               = True    
    
    return rotor 

def rotor_blade_setup(rotor): 
    """ Defines a dummy vehicle for rotor blade optimization.
          
          Inputs:  
             rotor   - rotor data structure                  [None] 
              
          Outputs:  
             configs - configuration used in optimization    [None]
              
          Assumptions: 
             N/A 
        
          Source:
             None
    """    
    vehicle                             = SUAVE.Vehicle()  
    net                                 = SUAVE.Components.Energy.Networks.Battery_Propeller()
    net.number_of_lift_rotor_engines    = 1
    net.identical_lift_rotors           = True  
    net.lift_rotors.append(rotor)  
    vehicle.append_component(net) 
    configs                             = SUAVE.Components.Configs.Config.Container() 
    base_config                         = SUAVE.Components.Configs.Config(vehicle) 
    config                              = SUAVE.Components.Configs.Config(base_config)
    config.tag                          = 'rotor_testbench'
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
    """ Modifies geometry of rotor blade 
          
          Inputs:  
             nexus     - SUAVE optmization framework with rotor blade data structure [None]
              
          Outputs:  
             procedure - optimization methodology                                    [None]
              
          Assumptions: 
             N/A 
        
          Source:
             None
    """        
    # Pull out the vehicles
    vehicle = nexus.vehicle_configurations.rotor_testbench 
    rotor   = vehicle.networks.battery_propeller.lift_rotors.rotor 
    
    # Update geometry of blade
    c       = updated_blade_geometry(rotor.radius_distribution/rotor.tip_radius ,rotor.chord_r,rotor.chord_p,rotor.chord_q,rotor.chord_t)     
    beta    = updated_blade_geometry(rotor.radius_distribution/rotor.tip_radius ,rotor.twist_r,rotor.twist_p,rotor.twist_q,rotor.twist_t)   
    
    rotor.chord_distribution               = c
    rotor.twist_distribution               = beta  
    rotor.mid_chord_alignment              = c/4. - c[0]/4.
    airfoil_geometry_data                  = import_airfoil_geometry(rotor.airfoil_geometry) 
    t_max                                  = np.take(airfoil_geometry_data.max_thickness,rotor.airfoil_polar_stations,axis=0)*c
    rotor.airfoil_data                     = airfoil_geometry_data
    rotor.max_thickness_distribution       = t_max   
     
    # diff the new data
    vehicle.store_diff() 
    
    return nexus    

def updated_blade_geometry(chi,c_r,p,q,c_t):
    """ Computes planform function of twist and chord distributron using hyperparameters  
          
          Inputs:  
             chi - rotor radius distribution [None]
             c_r - hyperparameter no. 1      [None]
             p   - hyperparameter no. 2      [None]
             q   - hyperparameter no. 3      [None]
             c_t - hyperparameter no. 4      [None]
              
          Outputs:  
             x_lin  - function distribution  [None]
              
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
    """ Performs aerodynamic and aeroacoustic analysis on rotor blade, computes constraint
        violations and objective function.
          
          Inputs:  
             nexus     - SUAVE optmization framework with rotor blade data structure [None]
              
          Outputs:  
             nexus     - SUAVE optmization framework with rotor blade data structure [None]
              
          Assumptions: 
             N/A 
        
          Source:
             N/A
    """    
    summary       = nexus.summary 
    vehicle       = nexus.vehicle_configurations.rotor_testbench  
    lift_rotors   = vehicle.networks.battery_propeller.lift_rotors
    
    # -------------------------------------------------------
    # RUN AEROACOUSTICS MODELS
    # -------------------------------------------------------    
    # unpack rotor properties 
    rotor         = lift_rotors.rotor 
    c             = rotor.chord_distribution 
    omega         = rotor.angular_velocity 
    V             = rotor.freestream_velocity   
    alt           = rotor.design_altitude
    alpha         = rotor.optimization_parameters.aeroacoustic_weight
    epsilon       = rotor.optimization_parameters.slack_constaint 
    ideal_SPL     = rotor.optimization_parameters.ideal_SPL_dBA  
    
    # Calculate atmospheric properties
    atmosphere     = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data      = atmosphere.compute_values(alt)   
    T              = atmo_data.temperature[0]
    rho            = atmo_data.density[0]
    a              = atmo_data.speed_of_sound[0]
    mu             = atmo_data.dynamic_viscosity[0]  

    # Define microphone locations
    theta     = np.array([45,90,135])*Units.degrees + 1E-1
    S         = 10. 
    ctrl_pts  = 1 
    positions = np.zeros(( len(theta),3))
    for i in range(len(theta)):
        if theta[i]*Units.degrees < np.pi/2:
            positions[i][:] = [-S*np.cos(theta[i]*Units.degrees)  ,S*np.sin(theta[i]*Units.degrees), 0.0]
        else: 
            positions[i][:] = [S*np.sin(theta[i]*Units.degrees- np.pi/2)  ,S*np.cos(theta[i]*Units.degrees - np.pi/2), 0.0] 

    # Define run conditions 
    rotor.inputs.omega                               = np.atleast_2d(omega).T
    conditions                                       = Aerodynamics()   
    conditions.freestream.density                    = np.ones((ctrl_pts,1)) * rho
    conditions.freestream.dynamic_viscosity          = np.ones((ctrl_pts,1)) * mu
    conditions.freestream.speed_of_sound             = np.ones((ctrl_pts,1)) * a 
    conditions.freestream.temperature                = np.ones((ctrl_pts,1)) * T 
    conditions.frames.inertial.velocity_vector       = np.array([[V, 0. ,0.]])
    conditions.propulsion.throttle                   = np.ones((ctrl_pts,1))*1.0
    conditions.frames.body.transform_to_inertial     = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]])

    # Run Propeller model 
    thrust , torque, power, Cp  , noise_data , etap  = rotor.spin(conditions) 

    # Prepare Inputs for Noise Model  
    conditions.noise.total_microphone_locations      = np.repeat(positions[ np.newaxis,:,: ],1,axis=0)
    conditions.aerodynamics.angle_of_attack          = np.ones((ctrl_pts,1))* 0. * Units.degrees 
    segment                                          = Segment() 
    segment.state.conditions                         = conditions
    segment.state.conditions.expand_rows(ctrl_pts) 

    # Store Noise Data 
    noise                                            = SUAVE.Analyses.Noise.Fidelity_One() 
    settings                                         = noise.settings   
    num_mic                                          = len(conditions.noise.total_microphone_locations[0])  
    conditions.noise.number_of_microphones           = num_mic
    
    # Run noise model    
    if alpha != 1: 
        propeller_noise  = propeller_mid_fidelity(lift_rotors,noise_data,segment,settings)   
        mean_SPL         = np.mean(propeller_noise.SPL_dBA) 
        Acoustic_Metric  = mean_SPL 
    else:
        Acoustic_Metric  = 0 
        mean_SPL         = 0
   
    # -------------------------------------------------------
    # CONTRAINTS
    # -------------------------------------------------------
    # thrust/power constraint
    if rotor.design_thrust == None:
        summary.thrust_power_residual = epsilon*rotor.design_power - abs(power[0][0] - rotor.design_power)
        ideal_aero                    = (rotor.design_power/V)
        Aerodynamic_Metric            = thrust[0][0]
    else: 
        summary.thrust_power_residual = epsilon*rotor.design_thrust - abs(thrust[0][0] - rotor.design_thrust)
        ideal_aero                    = rotor.design_thrust*V
        Aerodynamic_Metric            = power[0][0]     

    # q to p ratios 
    summary.chord_p_to_q_ratio = rotor.chord_p/rotor.chord_q
    summary.twist_p_to_q_ratio = rotor.twist_p/rotor.twist_q
    
    # Cl constraint  
    mean_CL = np.mean(noise_data.lift_coefficient[0])
    
    # blade taper consraint 
    blade_taper = c[-1]/c[0]
    summary.blade_taper_constraint_1  = blade_taper 
    summary.blade_taper_constraint_2  = blade_taper

    # -------------------------------------------------------
    # OBJECTIVE FUNCTION
    # -------------------------------------------------------     
    summary.Aero_Acoustic_Obj =  LA.norm((Aerodynamic_Metric - ideal_aero)/ideal_aero)*alpha \
                                + LA.norm((Acoustic_Metric - ideal_SPL)/ideal_SPL)*(1-alpha)
        
    # -------------------------------------------------------
    # PRINT ITERATION PERFOMRMANCE
    # -------------------------------------------------------                
    print("Aero_Acoustic_Obj       : " + str(summary.Aero_Acoustic_Obj)) 
    if rotor.design_thrust == None: 
        print("Power                   : " + str(power[0][0])) 
    if rotor.design_power == None: 
        print("Thrust                  : " + str(thrust[0][0]))   
    print("Average SPL             : " + str(mean_SPL))  
    print("Thrust/Power Residual   : " + str(summary.thrust_power_residual)) 
    print("Blade Taper             : " + str(blade_taper))
    print("Mean CL                 : " + str(mean_CL))  
    print("\n\n") 
    
    return nexus 