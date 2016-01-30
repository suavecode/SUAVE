import numpy as np


# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------
def expand_state(segment,state):
    
    # unpack
    climb_angle  = segment.climb_angle
    air_speed    = segment.air_speed   
    conditions   = state.conditions
    
    #Necessary input for determination of noise trajectory    
    dt = 0.5  #time step in seconds for noise calculation - Certification requirement    
    x0 = 6500 #Position of the Flyover microphone relatively to the break-release point
    
    # process velocity vector
    v_x=air_speed*np.cos(climb_angle)
    
    #number of time steps (space discretization)
    total_time=(x0+500)/v_x    
    n_points   = np.ceil(total_time/dt +1)       
    
    state.numerics.number_control_points = n_points
    
    state.expand_rows(n_points)      
    
    return

def initialize_conditions(segment,state):
    
    dt=0.5  #time step in seconds for noise calculation
    
    # unpack
    climb_angle = segment.climb_angle
    air_speed   = segment.air_speed   
    t_nondim    = state.numerics.dimensionless.control_points
    conditions  = state.conditions  
    
    # process velocity vector
    v_mag = air_speed
    v_x   = v_mag * np.cos(climb_angle)
    v_z   = -v_mag * np.sin(climb_angle)    

    #initial altitude
    alt0 = 10.668   #(35ft)
    altf = alt0 + (-v_z)*dt*len(t_nondim)

    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0    
    
    # pack conditions    
    conditions.frames.inertial.velocity_vector[:,0] = v_x
    conditions.frames.inertial.velocity_vector[:,2] = v_z
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context
