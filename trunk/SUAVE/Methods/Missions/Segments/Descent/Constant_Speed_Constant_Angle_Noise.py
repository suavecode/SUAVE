# Constant_Speed_Constant_Angle_noise.py
# 
# Created:  Nov 2015, C. Ilario
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Expand State
# ----------------------------------------------------------------------

def expand_state(segment,state):

    
    #Modification 11/04:
    #Necessary input for determination of noise trajectory    
    dt = 0.5  #time step in seconds for noise calculation - Certification requirement    
    
    # unpack
    descent_angle = segment.descent_angle
    air_speed     = segment.air_speed   
    conditions    = state.conditions      
    
    # process velocity vector
    s0 = 4000. #Defining the initial position of the measureament will start at 4 km from the threshold
    v_x           = air_speed * np.cos(segment.descent_angle) 
    
    #number of time steps (space discretization)  
    total_time    = s0/v_x    
    n_points      = np.ceil(total_time/dt +1)       
    
    state.numerics.number_control_points = n_points
    
    state.expand_rows(n_points)      
    
    return

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

def initialize_conditions(segment,state):
    
    # unpack
    descent_angle= segment.descent_angle
    air_speed    = segment.air_speed   
    t_nondim     = state.numerics.dimensionless.control_points
    conditions   = state.conditions  
    
    altf = 50. * Units.feet #(50ft last point for the noise measureament)
    
    #Linear equation: y-y0=m(x-x0)
    m_xx0 = 2000 * np.tan(descent_angle)
    y0    =  m_xx0 + altf  #(Altitude at the microphone X position)
    
    alt0 = y0 + m_xx0 #(Initial altitude of the aircraft)

    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0
    
    # process velocity vector
    v_mag = air_speed
    v_x   = v_mag * np.cos(-descent_angle)
    v_z   = -v_mag * np.sin(-descent_angle)
    
    # pack conditions    
    conditions.frames.inertial.velocity_vector[:,0] = v_x
    conditions.frames.inertial.velocity_vector[:,2] = v_z
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context    
    
