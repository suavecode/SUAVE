# Optimized.py
# 
# Created:  Dec 2016, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

def unpack_unknowns(segment,state):
    
    # unpack unknowns and givens
    throttle = state.unknowns.throttle
    theta    = state.unknowns.body_angle
    gamma1   = state.unknowns.flight_path_angle
    vel      = state.unknowns.velocity
    alt0     = segment.altitude_start
    altf     = segment.altitude_end
    vel0     = segment.air_speed_start
    velf     = segment.air_speed_end 

    # Overide the speeds   
    v_mag = np.concatenate([[[vel0]],vel,[[velf]]])
    gamma = np.concatenate([[[0]],gamma1,[[0]]])
    
    # process velocity vector
    v_x   =  v_mag * np.cos(gamma)
    v_z   = -v_mag * np.sin(gamma)    

    # apply unknowns and pack conditions   
    state.conditions.propulsion.throttle[:,0]             = throttle[:,0]
    state.conditions.frames.body.inertial_rotations[:,1]  = theta[:,0]   
    state.conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0] 
    state.conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0] 

def initialize_unknowns(segment,state):
    
    # unpack unknowns and givens
    gamma    = state.unknowns.flight_path_angle
    vel      = state.unknowns.velocity 
    v0       = segment.air_speed_start
    ones_m2  = state.ones_row_m2(1)
    ones     = state.ones_row(1)
    
    # repack
    state.unknowns.velocity          = v0*ones_m2
    state.unknowns.flight_path_angle = gamma[0]*ones_m2
    
def update_differentials(segment,state):

    # unpack
    numerics   = state.numerics
    conditions = state.conditions
    x    = numerics.dimensionless.control_points
    D    = numerics.dimensionless.differentiate
    I    = numerics.dimensionless.integrate 
    r    = state.conditions.frames.inertial.position_vector
    v    = state.conditions.frames.inertial.velocity_vector
    alt0 = segment.altitude_start
    altf = segment.altitude_end    

    dz = altf - alt0
    vz = -v[:,2,None] # maintain column array

    # get overall time step
    dt = (dz/np.dot(I,vz))[-1]

    # rescale operators
    x = x * dt
    D = D / dt
    I = I * dt
    
    # Calculate the altitudes
    alt = np.dot(I,vz) + segment.altitude_start
    
    # pack
    t_initial                                       = state.conditions.frames.inertial.time[0,0]
    numerics.time.control_points                    = x
    numerics.time.differentiate                     = D
    numerics.time.integrate                         = I
    conditions.frames.inertial.time[:,0]            = t_initial + x[:,0] 
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context    

    return

def objective(segment,state):
    
    if segment.objective is not None:
        if segment.minimize ==True:
            objective = eval('state.'+segment.objective)
        else:
            objective = -eval('state.'+segment.objective)
    else:
        objective = 0.
        
    state.objective_value = objective
        

def constraints(segment,state):
    
    # Residuals
    state.constraint_values = state.residuals.pack_array()
        

def cache_inputs(segment,state):
    state.inputs_last = state.unknowns.pack_array()