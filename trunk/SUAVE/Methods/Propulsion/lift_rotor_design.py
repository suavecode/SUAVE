## @ingroup Methods-Propulsion
# lift_rotor_design.py 
#
# Created: Feb 2022, M. Clarke

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  
# SUAVE Imports 
import SUAVE 
from SUAVE.Core                                                                                import Units, Data  
from SUAVE.Analyses.Mission.Segments.Segment                                                   import Segment 
from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_mid_fidelity                         import propeller_mid_fidelity
import SUAVE.Optimization.Package_Setups.scipy_setup                                           as scipy_setup
from SUAVE.Optimization                                                                        import Nexus      
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_properties   import compute_airfoil_properties
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series         import compute_naca_4series
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry      import import_airfoil_geometry
from SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics                                   import Aerodynamics 
from SUAVE.Analyses.Process                                                                    import Process   

# Python package imports  
from numpy import linalg as LA  
import numpy as np 
import scipy as sp 
import time 

# ----------------------------------------------------------------------
#  Rotor Design
# ----------------------------------------------------------------------
## @ingroup Methods-Propulsion
def lift_rotor_design(rotor,number_of_stations = 20,solver_name= 'SLSQP',
                      solver_sense_step = 1E-5,solver_tolerance = 1E-4,print_iterations = False):   
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
          Twist distribution                   [array of radians]
          Chord distribution                   [array of meters]
              
          Assumptions: 
             N/A 
        
          Source:
             None 
    """    
    # Unpack rotor geometry  
    rotor_tag     = rotor.tag
    rotor.tag     = 'rotor'
    N             = number_of_stations       
    B             = rotor.number_of_blades    
    R             = rotor.tip_radius
    Rh            = rotor.hub_radius 
    design_thrust = rotor.design_thrust
    design_power  = rotor.design_power
    chi0          = Rh/R  
    chi           = np.linspace(chi0,1,N+1)  
    chi           = chi[0:N]  
    airfoils      = rotor.Airfoils      
    a_loc         = rotor.airfoil_polar_stations    
    
    # determine target values 
    if (design_thrust == None) and (design_power== None):
        raise AssertionError('Specify either design thrust or design power!') 
    elif (design_thrust!= None) and (design_power!= None):
        raise AssertionError('Specify either design thrust or design power!') 
    if rotor.rotation == None:
        rotor.rotation = list(np.ones(int(B)))   
        
    num_airfoils = len(airfoils.keys())
    if num_airfoils>0:
        if len(a_loc) != N:
            raise AssertionError('\nDimension of airfoil sections must be equal to number of stations on propeller') 
        
        for _,airfoil in enumerate(airfoils):  
            if airfoil.geometry == None: # first, if airfoil geometry data not defined, import from geoemtry files
                if airfoil.NACA_4_series_flag: # check if naca 4 series of airfoil from datafile
                    airfoil.geometry = compute_naca_4series(airfoil.coordinate_file,airfoil.number_of_points)
                else:
                    airfoil.geometry = import_airfoil_geometry(airfoil.coordinate_file,airfoil.number_of_points) 
    
            if airfoil.polars == None: # compute airfoil polars for airfoils
                airfoil.polars = compute_airfoil_properties(airfoil.geometry, airfoil_polar_files= airfoil.polar_files) 
                     
    # thickness to chord         
    t_c           = np.zeros(N)    
    if num_airfoils>0:
        for j,airfoil in enumerate(airfoils): 
            a_geo         = airfoil.geometry
            locs          = np.where(np.array(a_loc) == j ) 
            t_c[locs]     = a_geo.thickness_to_chord
            
    # append additional rotor properties for optimization  
    rotor.number_of_blades                 = int(B)  
    rotor.thickness_to_chord               = t_c
    rotor.radius_distribution              = chi*R      
    
    # assign intial conditions for twist and chord distribution functions
    rotor.chord_r  = rotor.radius_distribution[0] 
    rotor.chord_p  = 1.0       
    rotor.chord_q  = 0.5       
    rotor.chord_t  = 0.05*R    
    rotor.twist_r  = np.pi/6   
    rotor.twist_p  = 1.0       
    rotor.twist_q  = 0.5       
    rotor.twist_t  = np.pi/10   
    
    # start optimization 
    ti                   = time.time()   
    optimization_problem = rotor_optimization_setup(rotor,print_iterations )  
    output               = scipy_setup.SciPy_Solve(optimization_problem,solver=solver_name, sense_step = solver_sense_step,tolerance = solver_tolerance)    
    tf                   = time.time()
    elapsed_time         = round((tf-ti)/60,2)
    print('Lift-rotor Optimization Simulation Time: ' + str(elapsed_time) + ' mins')   
    
    # print optimization results 
    print (output)  
    
    # set remaining rotor variables using optimized parameters 
    rotor     = set_optimized_rotor_planform(rotor,optimization_problem)
    rotor.tag = rotor_tag
    
    return rotor
  
def rotor_optimization_setup(rotor,print_iterations ):
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
    tm_ll  = rotor.optimization_parameters.tip_mach_range[0]
    tm_ul  = rotor.optimization_parameters.tip_mach_range[1]    
    tm_0   = (tm_ul + tm_ll)/2
    inputs = []
    inputs.append([ 'tip_mach'   , tm_0      , tm_ll  , tm_ul    , 1.0     ,  1*Units.less])
    inputs.append([ 'chord_r'    , 0.1*R     , 0.05*R , 0.2*R    , 1.0     ,  1*Units.less])
    inputs.append([ 'chord_p'    , 2         , 0.25   , 2.0      , 1.0     ,  1*Units.less])
    inputs.append([ 'chord_q'    , 1         , 0.25   , 1.5      , 1.0     ,  1*Units.less])
    inputs.append([ 'chord_t'    , 0.05*R    , 0.02*R , 0.1*R    , 1.0     ,  1*Units.less])  
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
    constraints.append([ 'thrust_power_residual'    ,  '>'  ,  0.0 ,   1.0   , 1*Units.less])  
    constraints.append([ 'blade_taper_constraint_1' ,  '>'  ,  0.3 ,   1.0   , 1*Units.less])  
    constraints.append([ 'blade_taper_constraint_2' ,  '<'  ,  0.9 ,   1.0   , 1*Units.less])
    constraints.append([ 'blade_twist_constraint'   ,  '>'  ,  0.0 ,   1.0   , 1*Units.less])
    constraints.append([ 'max_sectional_cl'         ,  '<'  ,  0.8 ,   1.0   , 1*Units.less])
    constraints.append([ 'chord_p_to_q_ratio'       ,  '>'  ,  0.5 ,   1.0   , 1*Units.less])    
    constraints.append([ 'twist_p_to_q_ratio'       ,  '>'  ,  0.5 ,   1.0   , 1*Units.less])   
    problem.constraints =  np.array(constraints,dtype=object)                
    
    # -------------------------------------------------------------------
    #  Aliases
    # ------------------------------------------------------------------- 
    aliases = []
    aliases.append([ 'tip_mach'                  , 'vehicle_configurations.*.networks.battery_propeller.propellers.rotor.design_tip_mach' ])
    aliases.append([ 'chord_r'                   , 'vehicle_configurations.*.networks.battery_propeller.propellers.rotor.chord_r' ])
    aliases.append([ 'chord_p'                   , 'vehicle_configurations.*.networks.battery_propeller.propellers.rotor.chord_p' ])
    aliases.append([ 'chord_q'                   , 'vehicle_configurations.*.networks.battery_propeller.propellers.rotor.chord_q' ])
    aliases.append([ 'chord_t'                   , 'vehicle_configurations.*.networks.battery_propeller.propellers.rotor.chord_t' ]) 
    aliases.append([ 'twist_r'                   , 'vehicle_configurations.*.networks.battery_propeller.propellers.rotor.twist_r' ])
    aliases.append([ 'twist_p'                   , 'vehicle_configurations.*.networks.battery_propeller.propellers.rotor.twist_p' ])
    aliases.append([ 'twist_q'                   , 'vehicle_configurations.*.networks.battery_propeller.propellers.rotor.twist_q' ])
    aliases.append([ 'twist_t'                   , 'vehicle_configurations.*.networks.battery_propeller.propellers.rotor.twist_t' ]) 
    aliases.append([ 'Aero_Acoustic_Obj'         , 'summary.Aero_Acoustic_Obj'       ])  
    aliases.append([ 'thrust_power_residual'     , 'summary.thrust_power_residual'   ]) 
    aliases.append([ 'blade_taper_constraint_1'  , 'summary.blade_taper_constraint_1'])  
    aliases.append([ 'blade_taper_constraint_2'  , 'summary.blade_taper_constraint_2'])   
    aliases.append([ 'blade_twist_constraint'    , 'summary.blade_twist_constraint'])   
    aliases.append([ 'max_sectional_cl'          , 'summary.max_sectional_cl'])  
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
    nexus.print_iterations  = print_iterations 
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
             Default noise measurements 135 degrees from rotor plane 
        
          Source:
             None
    """    
    network                  = optimization_problem.vehicle_configurations.rotor_testbench.networks.battery_propeller
    rotor_opt                = network.propellers.rotor 
    r                        = rotor.radius_distribution
    R                        = rotor.tip_radius      
    B                        = rotor.number_of_blades  
    rotor.design_tip_mach    = rotor_opt.design_tip_mach
    V                        = rotor.freestream_velocity  
    alt                      = rotor.design_altitude 
    theta                    = rotor.optimization_parameters.microphone_angle    
    rotor.chord_distribution = rotor_opt.chord_distribution
    rotor.twist_distribution = rotor_opt.twist_distribution 
    c                        = rotor.chord_distribution
    airfoils                 = rotor.Airfoils      
    a_loc                    = rotor.airfoil_polar_stations     
  
    # Calculate atmospheric properties
    atmosphere                = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data                 = atmosphere.compute_values(alt)    
    ctrl_pts                  = 1  
    omega                     = rotor.design_tip_mach*atmo_data.speed_of_sound[0][0]/rotor.tip_radius 

    # microphone locations
    S                         = np.maximum(alt , 20*Units.feet)  
    positions                 = np.array([[0.0 , S*np.sin(theta)  ,S*np.cos(theta)]])     

    # Set up for Propeller Model
    rotor.inputs.omega                                     = np.atleast_2d(omega).T
    conditions                                             = Aerodynamics()   
    conditions.freestream.update(atmo_data)    
    conditions.frames.inertial.velocity_vector             = np.array([[0, 0. ,V]])
    conditions.propulsion.throttle                         = np.ones((ctrl_pts,1))*1.0
    conditions.frames.body.transform_to_inertial           = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., -1.]]]) 
    
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

    propeller_noise   = propeller_mid_fidelity(network.propellers,noise_data,segment,settings)   
    mean_SPL          = np.mean(propeller_noise.SPL_dBA) 

    if rotor.design_power == None: 
        rotor.design_power = power[0][0]
    if rotor.design_thrust == None: 
        rotor.design_thrust = -thrust[0][2]

    design_torque = power[0][0]/omega        
    blade_area    = sp.integrate.cumtrapz(B*c, r-r[0])
    sigma         = blade_area[-1]/(np.pi*R**2)   
    MCA           = c/4. - c[0]/4.   
    
    t_max         = np.zeros(len(c))    
    t_c           = np.zeros(len(c))    
    if len(airfoils.keys())>0:
        for j,airfoil in enumerate(airfoils): 
            a_geo         = airfoil.geometry
            locs          = np.where(np.array(a_loc) == j )
            t_max[locs]   = a_geo.max_thickness*c[locs] 
            t_c[locs]     = a_geo.thickness_to_chord  

    rotor.angular_velocity           = omega      
    rotor.design_torque              = design_torque
    rotor.max_thickness_distribution = t_max
    rotor.radius_distribution        = r 
    rotor.number_of_blades           = int(B) 
    rotor.design_power_coefficient   = Cp[0][0] 
    rotor.design_thrust_coefficient  = noise_data.thrust_coefficient[0][0] 
    rotor.mid_chord_alignment        = MCA
    rotor.thickness_to_chord         = t_c
    rotor.design_SPL_dBA             = mean_SPL
    rotor.design_performance         = noise_data
    rotor.design_acoustics           = propeller_noise
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
    net.number_of_propeller_engines    = 1
    net.identical_propellers           = True  
    net.propellers.append(rotor)  
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
    vehicle   = nexus.vehicle_configurations.rotor_testbench 
    rotor     = vehicle.networks.battery_propeller.propellers.rotor 
    airfoils  = rotor.Airfoils      
    a_loc     = rotor.airfoil_polar_stations   
    
    # Update geometry of blade
    c         = updated_blade_geometry(rotor.radius_distribution/rotor.tip_radius ,rotor.chord_r,rotor.chord_p,rotor.chord_q,rotor.chord_t)     
    beta      = updated_blade_geometry(rotor.radius_distribution/rotor.tip_radius ,rotor.twist_r,rotor.twist_p,rotor.twist_q,rotor.twist_t)   
  
    # compute max thickness distribution  
    t_max  = np.zeros(len(c))     
    if len(airfoils.keys())>0:
        for j,airfoil in enumerate(airfoils): 
            a_geo         = airfoil.geometry
            locs          = np.where(np.array(a_loc) == j )
            t_max[locs]   = a_geo.max_thickness*c[locs]   
    
    rotor.max_thickness_distribution = t_max
    rotor.chord_distribution         = c
    rotor.twist_distribution         = beta  
    rotor.mid_chord_alignment        = c/4. - c[0]/4. 
     
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
    summary        = nexus.summary 
    
    # -------------------------------------------------------
    # RUN AEROACOUSTICS MODELS
    # -------------------------------------------------------    
    # unpack rotor properties 
    vehicle        = nexus.vehicle_configurations.rotor_testbench  
    propellers     = vehicle.networks.battery_propeller.propellers
    rotor          = propellers.rotor 
    c              = rotor.chord_distribution 
    beta           = rotor.twist_distribution 
    V              = rotor.freestream_velocity     
    alt            = rotor.design_altitude
    theta          = rotor.optimization_parameters.microphone_angle 
    alpha          = rotor.optimization_parameters.aeroacoustic_weight
    epsilon        = rotor.optimization_parameters.slack_constaint 
    ideal_SPL      = rotor.optimization_parameters.ideal_SPL_dBA  
    print_iter     = nexus.print_iterations
    
    # Calculate atmospheric properties
    atmosphere     = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data      = atmosphere.compute_values(alt)    
    omega          = rotor.design_tip_mach*atmo_data.speed_of_sound[0]/rotor.tip_radius   

    # Define microphone locations 
    S              = np.maximum(alt , 20*Units.feet) 
    positions      = np.array([[0.0 , S*np.sin(theta)  ,S*np.cos(theta)]])
    ctrl_pts       = 1   

    # Define run conditions 
    rotor.inputs.omega                               = np.atleast_2d(omega).T
    conditions                                       = Aerodynamics()   
    conditions.freestream.update(atmo_data) 
    conditions.frames.inertial.velocity_vector       = np.array([[0, 0. ,V]])
    conditions.propulsion.throttle                   = np.ones((ctrl_pts,1))*1.0
    conditions.frames.body.transform_to_inertial     = np.array([[[1., 0., 0.],[0., 1., 0.],[0., 0., -1.]]])  

    # Run rotor model 
    thrust ,_, power, Cp  , noise_data , _ = rotor.spin(conditions) 

    # Set up noise model
    conditions.noise.total_microphone_locations      = np.repeat(positions[ np.newaxis,:,: ],1,axis=0)
    conditions.aerodynamics.angle_of_attack          = np.ones((ctrl_pts,1))* 0. * Units.degrees 
    segment                                          = Segment() 
    segment.state.conditions                         = conditions
    segment.state.conditions.expand_rows(ctrl_pts)  
    noise                                            = SUAVE.Analyses.Noise.Fidelity_One() 
    settings                                         = noise.settings   
    num_mic                                          = len(conditions.noise.total_microphone_locations[0])  
    conditions.noise.number_of_microphones           = num_mic
    
    # Run noise model if necessary    
    if alpha != 1:  
        try: 
            propeller_noise =  propeller_mid_fidelity(propellers,noise_data,segment,settings)    
            Acoustic_Metric = np.mean(propeller_noise.SPL_dBA) 
        except:
            Acoustic_Metric = 100        
    else:
        Acoustic_Metric  = 0 
   
    # -------------------------------------------------------
    # CONTRAINTS
    # -------------------------------------------------------
    # thrust/power constraint
    if rotor.design_thrust == None:
        summary.thrust_power_residual = epsilon*rotor.design_power - abs(power[0][0] - rotor.design_power)
    else: 
        summary.thrust_power_residual = epsilon*rotor.design_thrust - abs(-thrust[0][2] - rotor.design_thrust)

    # q to p ratios 
    summary.chord_p_to_q_ratio        = rotor.chord_p/rotor.chord_q
    summary.twist_p_to_q_ratio        = rotor.twist_p/rotor.twist_q
           
    # Cl constraint         
    summary.max_sectional_cl          = np.max(noise_data.lift_coefficient[0])
    mean_CL                           = np.mean(noise_data.lift_coefficient[0])
    
    # blade taper consraint 
    blade_taper                       =  c[-1]/c[0]
    summary.blade_taper_constraint_1  = blade_taper 
    summary.blade_taper_constraint_2  = blade_taper
     
    # blade twist consraint  
    summary.blade_twist_constraint    = beta[0] - beta[-1] 

    # figure of merit 
    C_t_UIUC                = noise_data.thrust_coefficient[0][0]
    C_t_rot                 = C_t_UIUC*8/(np.pi**3)
    C_p_UIUC                = Cp[0][0] 
    C_q_UIUC                = C_p_UIUC/(2*np.pi) 
    C_q_rot                 = C_q_UIUC*16/(np.pi**3)   
    C_p_rot                 = C_q_rot 
    ideal_FM                = 1
    FM                      = ((C_t_rot**1.5)/np.sqrt(2))/C_p_rot 
    summary.figure_of_merit = FM

    # -------------------------------------------------------
    # OBJECTIVE FUNCTION
    # -------------------------------------------------------
    summary.Aero_Acoustic_Obj =  LA.norm((FM - ideal_FM)*100/(ideal_FM*100))*alpha + \
        LA.norm((Acoustic_Metric - ideal_SPL)/(ideal_SPL))*(1-alpha) 
        
    # -------------------------------------------------------
    # PRINT ITERATION PERFOMRMANCE
    # -------------------------------------------------------  
    if print_iter:
        print("Aero_Acoustic_Obj       : " + str(summary.Aero_Acoustic_Obj))     
        print("Aero_Acoustic_Weight    : " + str(alpha))
        if rotor.design_thrust == None: 
            print("Thrust (N)              : " + str(-thrust[0][2]))   
            print("Power (kW)              : " + str(power[0][0]/1000)) 
        if rotor.design_power == None: 
            print("Thrust (N)              : " + str(-thrust[0][2]))   
            print("Power (kW)              : " + str(power[0][0]/1000)) 
        print("Tip Mach                : " + str(rotor.design_tip_mach))   
        print("Average SPL             : " + str(Acoustic_Metric))  
        print("Figure of Merit         : " + str(FM))  
        print("Thrust/Power Residual   : " + str(summary.thrust_power_residual)) 
        print("Blade Taper             : " + str(blade_taper))
        print("Max Sectional Cl        : " + str(summary.max_sectional_cl))  
        print("Blade CL                : " + str(mean_CL))  
        print("\n\n") 
    
    return nexus 