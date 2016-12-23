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
    alts     = state.unknowns.altitude 
    time     = state.unknowns.time
    alt0     = segment.altitude_start
    altf     = segment.altitude_end
    
    # Overide the altitudes and velocities
    alts[0]  = alt0
    alts[-1] = altf
    
    # dimensionalize time
    t_initial = state.conditions.frames.inertial.time[0,0]
    t_nondim  = state.numerics.dimensionless.control_points
    time      = t_nondim * (time) + t_initial    
    
    # apply unknowns and pack conditions   
    state.conditions.propulsion.throttle[:,0]             = throttle[:,0]
    state.conditions.frames.body.inertial_rotations[:,1]  = theta[:,0]   
    state.conditions.frames.inertial.position_vector[:,2] = -alts[:,0] # z points down
    state.conditions.frames.inertial.time[:,0]            = time[:,0]
    state.conditions.freestream.altitude[:,0]             = alts[:,0] # positive altitude in this context    
    
    
def calculate_velocities(segment,state):
    
    # unpack unknowns and givens
    throttle = state.unknowns.throttle
    theta    = state.unknowns.body_angle
    alts     = state.unknowns.altitude 
    vels     = state.unknowns.velocity 
    v0       = segment.air_speed_start
    vf       = segment.air_speed_end    
    D        = state.numerics.time.differentiate
    
    # Overide the velocities
    vels[0]  = v0
    vels[-1] = vf    
    
    # Figure out the altitudes
    alts = state.conditions.freestream.altitude
    
    # Calculate the rate of climb
    climb_rate = np.dot(D,alts)
    
    # process velocity vector
    v_mag = vels
    v_z   = climb_rate
    v_x   = np.sqrt( v_mag**2 - v_z**2 )
    
    #v_x[v_x>v_mag] = v_mag[v_x>v_mag]
    #v_x[np.isnan(v_x)] = v_mag[np.isnan(v_x)]
    
    #v_z[v_z>v_mag] = v_mag[v_z>v_mag]
    #v_z[np.isnan(v_z)] = v_mag[np.isnan(v_z)]
    
    # apply unknowns and pack conditions   
    state.conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
    state.conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0]  
    
def initialize_unknowns(segment,state):
    
    # unpack unknowns and givens
    alts     = state.unknowns.altitude 
    vels     = state.unknowns.velocity 
    alt0     = segment.altitude_start
    altf     = segment.altitude_end
    v0       = segment.air_speed_start
    vf       = segment.air_speed_end
    
    # Overide the altitudes and velocities
    alts = np.reshape(np.linspace(alt0,altf,np.size(alts)),np.shape(alts))
    vels = np.reshape(np.linspace(v0,vf,np.size(vels)),np.shape(vels))
    
    # repack
    state.unknowns.velocity = vels
    state.unknowns.altitude = alts
    

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
        
    state.constraint_values = state.residuals.pack_array()
        

def cache_inputs(segment,state):
    state.inputs_last = state.unknowns.pack_array()