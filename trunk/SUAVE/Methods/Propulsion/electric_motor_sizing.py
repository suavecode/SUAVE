## @ingroup Methods-Propulsion
# electric_motor_sizing.py
# 
# Created:  Jan 2016, E. Botero
# Modified: Feb 2020, M. Clarke 
#           Mar 2020, M. Clarke
#           Sep 2020, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE

# package imports
import numpy as np
from scipy.optimize import minimize 
from SUAVE.Core import Units 
# ----------------------------------------------------------------------
#  size_from_kv
# ----------------------------------------------------------------------
## @ingroup Methods-Propulsion
def size_from_kv(motor):
    """
    Determines a motors mass based on the speed constant KV
    
    Source:
    Gur, O., Rosen, A, AIAA 2008-5916.
    
    Inputs:
    motor    (to be modified)
      kv       motor speed constant
    
    Outputs:
    motor.
      resistance         [ohms]
      no_load_current    [amps]
      mass               [kg]
    
    
    """
    
    # Set the KV     
    kv = motor.speed_constant 
    
    # Correlations from Gur:
    # Gur, O., Rosen, A, AIAA 2008-5916. 
    
    B_KV = 50.   * Units['rpm*kg/volt']
    B_RA = 60000.* Units['(rpm**2)*ohm/(volt**2)']
    B_i0 = 0.2   * Units['amp*(ohm**0.6)']
    
    # Do the calculations from the regressions
    mass = B_KV/kv
    res  = B_RA/(kv**2.)
    i0   = B_i0/(res**0.6)
    
    # pack
    motor.resistance           = res
    motor.no_load_current      = i0
    motor.mass_properties.mass = mass
    
    return motor

## @ingroup Methods-Propulsion
def size_from_mass(motor):
    """
    Sizes motor from mass
    
    Source:
    Gur, O., Rosen, A, AIAA 2008-5916.
    
    Inputs:
    motor.    (to be modified)
      mass               [kg]
    
    Outputs:
    motor.
      resistance         [ohms]
      no_load_current    [amps] 
    """     
    mass = motor.mass_properties.mass 
    
    # Correlations from Gur:
    # Gur, O., Rosen, A, AIAA 2008-5916.  
    
    B_KV = 50.   * Units['rpm*kg/volt']
    B_RA = 60000.* Units['(rpm**2)*ohm/(V**2)']
    B_i0 = 0.2   * Units['amp*(ohm**0.6)']
    
    # Do the calculations from the regressions
    kv   = B_KV/mass
    res  = B_RA/(kv**2.)
    i0   = B_i0/(res**0.6) 

    # Unpack the motor
    motor.resistance      = res 
    motor.no_load_current = i0  
    motor.speed_constant  = kv    

    return motor 

## @ingroup Methods-Propulsion
def size_optimal_motor(motor,prop):
    ''' Optimizes the motor to obtain the best combination of speed constant and resistance values
    by essentially sizing the motor for a design RPM value. Note that this design RPM 
    value can be compute from design tip mach  
    
    Assumptions:
    motor design performance occurs at 90% nominal voltage to account for off design conditions 
    
    Source:
    N/A
    
    Inputs:
    prop.
      design_torque          [Nm]
      angular_velocity       [rad/s]
      origin                 [m]
      
    motor.     
      no_load_current        [amps]
      mass_properties.mass   [kg]
      
    Outputs:
    motor.
      speed_constant         [untiless]
      design_torque          [Nm] 
      motor.resistance       [Ohms]
      motor.angular_velocity [rad/s]
      motor.origin           [m]
    '''    
    
    # assign propeller radius
    motor.propeller_radius      = prop.tip_radius
   
    # append motor locations based on propeller locations 
    motor.origin                = prop.origin  
    
    # motor design torque 
    motor.design_torque         = prop.design_torque  
    
    # design conditions for motor 
    io                          = motor.no_load_current
    v                           = motor.nominal_voltage
    omeg                        = prop.angular_velocity/motor.gear_ratio    
    etam                        = motor.efficiency 
    Q                           = motor.design_torque 
    
    # motor design rpm 
    motor.angular_velocity      = omeg
    
    # solve for speed constant   
    opt_params = optimize_kv(io, v , omeg,  etam ,  Q)
    
    Kv  =  opt_params[0]
    Res =  opt_params[1]    
    
    motor.speed_constant   = Kv 
    motor.resistance       = Res 
    
    return motor 
  
## @ingroup Methods-Propulsion
def optimize_kv(io, v , omeg,  etam ,  Q, kv_lower_bound =  0.01, Res_lower_bound = 0.001, kv_upper_bound = 100, Res_upper_bound = 10 ): 
    ''' Optimizer for compute_optimal_motor_parameters function  
    
    Source:
    N/A
    
    Inputs:
    motor    (to be modified)
    
    Outputs:
    motor.
      speed_constant     [untiless]
      no_load_current    [amps]
    '''        
    # objective  
    
    args = (v , omeg,  etam , Q , io )
    
    hard_cons = [{'type':'eq', 'fun': hard_constraint_1,'args': args},
                 {'type':'eq', 'fun': hard_constraint_2,'args': args}]
    
    slack_cons = [{'type':'eq', 'fun': slack_constraint_1,'args': args},
                  {'type':'eq', 'fun': slack_constraint_2,'args': args}] 
  
    torque_con = [{'type':'eq', 'fun': hard_constraint_2,'args': args}]
    
    bnds = ((kv_lower_bound, kv_upper_bound), (Res_lower_bound , Res_upper_bound))
    
    # try hard constraints to find optimum motor parameters
    sol = minimize(objective, [0.5, 0.1], args=(v , omeg,  etam , Q , io) , method='SLSQP', bounds=bnds, tol=1e-6, constraints=hard_cons) 
    
    if sol.success == False:
        # use slack constraints  if optimum motor parameters cannot be found 
        print('\n Optimum motor design failed. Using slack constraints')
        sol = minimize(objective, [0.5, 0.1], args=(v , omeg,  etam , Q , io) , method='SLSQP', bounds=bnds, tol=1e-6, constraints=slack_cons)
        
        # use one constraints as last resort if optimum motor parameters cannot be found 
        if sol.success == False:
            print ('\n Slack contraints failed. Using one constraint')
            sol = minimize(objective, [10], args=(v , omeg,  etam , Q , io) , method='SLSQP', bounds=bnds, tol=1e-6, constraints= torque_con)
    
    return sol.x   
  
# objective function
def objective(x, v , omeg,  etam , Q , io ): 
    return (v - omeg/x[0])/x[1]   

# hard efficiency constraint
def hard_constraint_1(x, v , omeg,  etam , Q , io ): 
    return etam - (1- (io*x[1])/(v - omeg/x[0]))*(omeg/(v*x[0]))   

# hard torque equality constraint
def hard_constraint_2(x, v , omeg,  etam , Q , io ): 
    return ((v - omeg/x[0])/x[1] - io)/x[0] - Q  

# slack efficiency constraint 
def slack_constraint_1(x, v , omeg,  etam , Q , io ): 
    return abs(etam - (1- (io*x[1])/(v - omeg/x[0]))*(omeg/(v*x[0]))) - 0.2

# slack torque equality constraint 
def slack_constraint_2(x, v , omeg,  etam , Q , io ): 
    return  abs(((v - omeg/x[0])/x[1] - io)/x[0] - Q) - 200 