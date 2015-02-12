import numpy as np
# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

def unpack_unknowns(segment,state):
    
    # unpack unknowns
    throttle = state.unknowns.throttle
    theta    = state.unknowns.body_angle
    
    # apply unknowns
    state.conditions.propulsion.throttle[:,0]            = throttle[:,0]
    state.conditions.frames.body.inertial_rotations[:,1] = theta[:,0]   
    
    # unpack conditions
    v = state.conditions.frames.inertial.velocity_vector
    D = state.numerics.time.differentiate

    # accelerations
    acc = np.dot(D,v)

    # pack conditions
    state.conditions.frames.inertial.acceleration_vector[:,:] = acc[:,:]    
    

# ----------------------------------------------------------------------
#  Residual Total Forces
# ----------------------------------------------------------------------

def residual_total_forces(segment,state):
    
    FT = state.conditions.frames.inertial.total_force_vector
    a  = state.conditions.frames.inertial.acceleration_vector
    m  = state.conditions.weights.total_mass    
    
    state.residuals.forces[:,0] = FT[:,0]/m[:,0] - a[:,0]
    state.residuals.forces[:,1] = FT[:,2]/m[:,0] - a[:,2]       

    return
       
def update_differentials_altitude(segment,state):
    """ Segment.update_differentials_altitude(conditions, numerics, unknowns)
        updates the differential operators t, D and I
        must return in dimensional time, with t[0] = 0

        Works with a segment discretized in vertical position, altitude

        Inputs - 
            unknowns      - data dictionary of segment free unknowns
            conditions    - data dictionary of segment conditions
            numerics - data dictionary of non-dimensional differential operators

        Outputs - 
            numerics - udpated data dictionary with dimensional numerics 

        Assumptions - 
            outputed operators are in dimensional time for the current solver iteration
            works with a segment discretized in vertical position, altitude

    """

    # unpack
    t = state.numerics.dimensionless.control_points
    D = state.numerics.dimensionless.differentiate
    I = state.numerics.dimensionless.integrate

    r = state.conditions.frames.inertial.position_vector
    v = state.conditions.frames.inertial.velocity_vector

    dz = r[-1,2] - r[0,2]
    vz = v[:,2,None] # maintain column array

    # get overall time step
    dt = np.dot( I[-1,:] * dz , 1/ vz[:,0] )

    # rescale operators
    D = D / dt
    I = I * dt
    t = t * dt

    # pack
    state.numerics.time.control_points = t
    state.numerics.time.differentiate = D
    state.numerics.time.integrate = I

    # time
    t_initial = state.conditions.frames.inertial.time[0,0]
    state.conditions.frames.inertial.time[:,0] = t_initial + t[:,0]

    return