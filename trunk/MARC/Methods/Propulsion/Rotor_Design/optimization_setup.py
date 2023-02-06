## @ingroup Methods-Propulsion-Rotor_Design
# optimization_setup.py 
#
# Created: Feb 2022, M. Clarke

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  
# MARC Imports  
import MARC 
from MARC.Core                                                    import Units, Data   
from MARC.Optimization                                            import Nexus       
from MARC.Methods.Propulsion.Rotor_Design.blade_geometry_setup    import blade_geometry_setup
from MARC.Methods.Propulsion.Rotor_Design.procedure_setup         import procedure_setup
# Python package imports   
import numpy as np  

## @ingroup Methods-Propulsion-Rotor_Design
def optimization_setup(rotor,number_of_stations,print_iterations):
    """ Sets up rotor optimization problem including design variables, constraints and objective function
        using MARC's Nexus optimization framework. Appends methodolody of planform modification to Nexus.
          Inputs: 
             rotor     - rotor data structure           [None]
             
          Outputs: 
              nexus    - MARC's optimization framework [None]
              
          Assumptions: 
            1) minimum allowable blade taper : 0.2  
            1) maximum allowable blade taper : 0.7     
        
          Source:
             None
    """    
    nexus                        = Nexus()
    problem                      = Data()
    nexus.optimization_problem   = problem
   
    if type(rotor) != MARC.Components.Energy.Converters.Prop_Rotor or  type(rotor) != MARC.Components.Energy.Converters.Lift_Rotor:
        assert('rotor must be of Lift-Rotor or Prop-Rotor class') 
        
    if type(rotor) == MARC.Components.Energy.Converters.Prop_Rotor:
        nexus.prop_rotor_flag = True 
    else:
        nexus.prop_rotor_flag = False 
    # -------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------  
    R         = rotor.tip_radius  
    tm_ll_h   = rotor.optimization_parameters.tip_mach_range[0]
    tm_ul_h   = rotor.optimization_parameters.tip_mach_range[1] 
    tm_0_h    = (tm_ul_h + tm_ll_h)/2 
    
    if nexus.prop_rotor_flag:    
        tm_ll_c          = rotor.optimization_parameters.tip_mach_range[0]
        tm_ul_c          = rotor.optimization_parameters.tip_mach_range[1]    
        
    inputs = []
    inputs.append([ 'chord_r'               ,  0.1*R    , 0.05*R     , 0.2*R     , 1.0     ,  1*Units.less])
    inputs.append([ 'chord_p'               ,  2        , 0.25       , 2.0       , 1.0     ,  1*Units.less])
    inputs.append([ 'chord_q'               ,  1        , 0.25       , 1.5       , 1.0     ,  1*Units.less])
    inputs.append([ 'chord_t'               ,  0.05*R   , 0.02*R     , 0.1*R     , 1.0     ,  1*Units.less])  
    inputs.append([ 'twist_r'               ,  np.pi/6  ,  0         , np.pi/4   , 1.0     ,  1*Units.less])
    inputs.append([ 'twist_p'               ,  1        , 0.25       , 2.0       , 1.0     ,  1*Units.less])
    inputs.append([ 'twist_q'               ,  0.5      , 0.25       , 1.5       , 1.0     ,  1*Units.less])
    inputs.append([ 'twist_t'               ,  np.pi/6  , 0          , np.pi/4   , 1.0     ,  1*Units.less])  
    inputs.append([ 'hover_tip_mach'        , tm_0_h    , tm_ll_h    , tm_ul_h   , 1.0     ,  1*Units.less])
    inputs.append([ 'OEI_tip_mach'          , tm_0_h    , tm_ll_h    , 0.85      , 1.0     ,  1*Units.less])  
    inputs.append([ 'OEI_collective_pitch'  , np.pi/6   , -np.pi/6   , np.pi/6   , 1.0      ,  1*Units.less]) 
    if nexus.prop_rotor_flag: 
        inputs.append([ 'cruise_tip_mach'         , tm_ll_c , tm_ll_c    , tm_ul_c   , 1.0     ,  1*Units.less]) 
        inputs.append([ 'cuise_collective_pitch'  ,-np.pi/6  , -np.pi/6   , np.pi/6   , 1.0     ,  1*Units.less]) 
    problem.inputs = np.array(inputs,dtype=object)   

    # -------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------- 
    problem.objective = np.array([  
                                 [  'objective'  ,  1.0   ,    1*Units.less] 
    ],dtype=object)
    
    # -------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------  
    constraints = [] 
    constraints.append([ 'hover_thrust_pow_res'      ,  '>'  ,  0.0 ,   1.0   , 1*Units.less])  
    constraints.append([ 'blade_taper_constraint_1'  ,  '>'  ,  0.3 ,   1.0   , 1*Units.less])  
    constraints.append([ 'blade_taper_constraint_2'  ,  '<'  ,  0.9 ,   1.0   , 1*Units.less])  
    constraints.append([ 'blade_twist_constraint'    ,  '>'  ,  0.0 ,   1.0   , 1*Units.less])
    constraints.append([ 'max_sectional_cl_hov'      ,  '<'  ,  0.8 ,   1.0   , 1*Units.less])
    constraints.append([ 'chord_p_to_q_ratio'        ,  '>'  ,  0.5 ,   1.0   , 1*Units.less])    
    constraints.append([ 'twist_p_to_q_ratio'        ,  '>'  ,  0.5 ,   1.0   , 1*Units.less])  
    constraints.append([ 'OEI_hov_thrust_pow_res'    ,  '>'  ,  0.0 ,   1.0   , 1*Units.less]) 
    if nexus.prop_rotor_flag:
        constraints.append([ 'cruise_thrust_pow_res'     ,  '>'  ,  0.0 ,   1.0   , 1*Units.less]) 
        constraints.append([ 'max_sectional_cl_cruise'   ,  '<'  ,  0.8 ,   1.0   , 1*Units.less])   
    problem.constraints =  np.array(constraints,dtype=object)                
    
    # -------------------------------------------------------------------
    #  Aliases
    # ------------------------------------------------------------------- 
    aliases = [] 
    aliases.append([ 'chord_r'                    , 'vehicle_configurations.*.networks.battery_electric_rotor.rotors.rotor.chord_r' ])
    aliases.append([ 'chord_p'                    , 'vehicle_configurations.*.networks.battery_electric_rotor.rotors.rotor.chord_p' ])
    aliases.append([ 'chord_q'                    , 'vehicle_configurations.*.networks.battery_electric_rotor.rotors.rotor.chord_q' ])
    aliases.append([ 'chord_t'                    , 'vehicle_configurations.*.networks.battery_electric_rotor.rotors.rotor.chord_t' ]) 
    aliases.append([ 'twist_r'                    , 'vehicle_configurations.*.networks.battery_electric_rotor.rotors.rotor.twist_r' ])
    aliases.append([ 'twist_p'                    , 'vehicle_configurations.*.networks.battery_electric_rotor.rotors.rotor.twist_p' ])
    aliases.append([ 'twist_q'                    , 'vehicle_configurations.*.networks.battery_electric_rotor.rotors.rotor.twist_q' ])
    aliases.append([ 'twist_t'                    , 'vehicle_configurations.*.networks.battery_electric_rotor.rotors.rotor.twist_t' ])     
    aliases.append([ 'hover_tip_mach'             , 'vehicle_configurations.hover.networks.battery_electric_rotor.rotors.rotor.hover.design_tip_mach' ]) 
    aliases.append([ 'objective'                  , 'summary.objective'       ])  
    aliases.append([ 'hover_thrust_pow_res'       , 'summary.hover_thrust_power_residual'   ]) 
    aliases.append([ 'blade_taper_constraint_1'   , 'summary.blade_taper_constraint_1'])   
    aliases.append([ 'blade_taper_constraint_2'   , 'summary.blade_taper_constraint_2'])   
    aliases.append([ 'blade_twist_constraint'     , 'summary.blade_twist_constraint'])    
    aliases.append([ 'max_sectional_cl_hov'       , 'summary.max_sectional_cl_hover'])   
    aliases.append([ 'chord_p_to_q_ratio'         , 'summary.chord_p_to_q_ratio'    ])  
    aliases.append([ 'twist_p_to_q_ratio'         , 'summary.twist_p_to_q_ratio'    ])   
    aliases.append([ 'OEI_hov_thrust_pow_res'     , 'summary.oei_thrust_power_residual'   ]) 
    aliases.append([ 'OEI_collective_pitch'       , 'vehicle_configurations.oei.networks.battery_electric_rotor.rotors.rotor.inputs.pitch_command' ]) 
    aliases.append([ 'OEI_tip_mach'               , 'vehicle_configurations.oei.networks.battery_electric_rotor.rotors.rotor.oei.design_tip_mach' ]) 
    if nexus.prop_rotor_flag: 
        aliases.append([ 'cruise_tip_mach'        , 'vehicle_configurations.cruise.networks.battery_electric_rotor.rotors.rotor.cruise.design_tip_mach' ])  
        aliases.append([ 'cuise_collective_pitch' , 'vehicle_configurations.cruise.networks.battery_electric_rotor.rotors.rotor.inputs.pitch_command' ])  
        aliases.append([ 'cruise_thrust_pow_res'  , 'summary.cruise_thrust_power_residual'   ]) 
        aliases.append([ 'max_sectional_cl_cruise', 'summary.max_sectional_cl_cruise'])  
         
    problem.aliases = aliases
    
    # -------------------------------------------------------------------
    #  Vehicles
    # -------------------------------------------------------------------
    nexus.vehicle_configurations = blade_geometry_setup(rotor,number_of_stations)
    
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
    nexus.procedure         = procedure_setup()
    
    # -------------------------------------------------------------------
    #  Summary
    # -------------------------------------------------------------------    
    nexus.summary        = Data()     
    nexus.results.hover  = Data() 
    nexus.results.cruise = Data()
    nexus.results.oei    = Data()
    
    return nexus   