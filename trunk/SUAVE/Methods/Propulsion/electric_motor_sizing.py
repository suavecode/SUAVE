## @ingroup Methods-Propulsion
# electric_motor_sizing.py
# 
# Created:  Jan 2016, E. Botero
# Modified: Feb 2020, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE

# package imports
import numpy as np
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
    start_kv             = 1
    end_kv               = 15     
    possible_kv_vals     = np.linspace(start_kv,end_kv,(end_kv-start_kv)*20 +1 , endpoint = True) * Units.rpm
    res_kv_vals          = ((v-omeg/possible_kv_vals)*(1.-etam*v*possible_kv_vals/omeg))/io  
    positive_res_vals    = np.extract(res_kv_vals > 0 ,res_kv_vals) 
    kv_idx               = np.where(res_kv_vals == min(positive_res_vals))[0][0]   
    kv                   = possible_kv_vals[kv_idx]  
    res                  = min(positive_res_vals) 
    
    motor.speed_constant = kv
    motor.resistance     = res 

    return motor