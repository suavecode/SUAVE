## @ingroup Methods-Propulsion
# electric_motor_sizing.py
# 
# Created:  Jan 2016, E. Botero
# Modified: 

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
def size_from_kv(motor,kv):
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
    motor.speed_constant = kv 
    
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


def size_from_mass(motor,mass):
    
    # Unpack the motor
    res  = motor.resistance
    i0   = motor.no_load_current
    kv   = motor.speed_constant

    # Set the KV    
    motor.mass_properties.mass = mass
    
    # Correlations from Gur:
    # Gur, O., Rosen, A, AIAA 2008-5916.  
    
    B_KV = 50.   * Units['rpm*kg/volt']
    B_RA = 60000.* Units['(rpm**2)*ohm/(V**2)']
    B_i0 = 0.2   * Units['amp*(ohm**0.6)']
    
    # Do the calculations from the regressions
    kv = B_KV/mass
    res  = B_RA/(kv**2.)
    i0   =  B_i0/(res**0.6)

    return motor
