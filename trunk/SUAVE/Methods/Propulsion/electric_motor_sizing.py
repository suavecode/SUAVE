## @ingroup Methods-Propulsion
# electric_motor_sizing.py
# 
# Created:  Jan 2016, E. Botero
# Modified: Feb 2020, M. Clarke 
#           Mar 2020, M. Clarke

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
    i0   =  B_i0/(res**0.6)
    
    # pack
    motor.resistance           = res
    motor.no_load_current      = i0
    motor.mass_properties.mass = mass
    
    return motor


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

def compute_optimal_motor_parameters(motor,prop):
    ''' Optimizes the motor to obtain the best combination of speed constant and resistance values
    by essentially you are sizing the motor for a design RPM value. Note that this design RPM 
    value can be compute from design tip mach  
    
    Source:
    N/A
    
    Inputs:
    motor    (to be modified)
    
    Outputs:
    motor.
      speed_constant     [untiless]
      no_load_current    [amps]
    '''    
    
    io                   = motor.no_load_current
    v                    = motor.nominal_voltage 
    omeg                 = prop.angular_velocity
    etam                 = motor.efficiency 
    motor.speed_constant = optimize_kv(io, v , omeg,  etam)
    motor.resistance     = ((v-omeg/motor.speed_constant)*(1.-etam*v*motor.speed_constant/omeg))/io
    
    return motor

def optimize_kv(io, v , omeg,  etam , lb = 0 , ub = 100): 
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
    objective = lambda x: ((v-omeg/x[0])*(1.-etam*v*x[0]/omeg))/io
    
    # bounds 
    bnds = [(lb,ub)]
    
    # constraints 
    cons = ({'type': 'ineq', 'fun': lambda x: ((v-omeg/x[0])*(1.-etam*v*x[0]/omeg))/io - 0.001}) # Added a tolerance on resistance, cant be less than 0.001 ohms  
    
    # solve 
    sol  = minimize(objective,(0.5), method = 'SLSQP',bounds = bnds, constraints = cons ) 

    return sol.x[0]  
 
 