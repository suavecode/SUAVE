import Common

def initialize_conditions(segment,state):

    Common.initialize_conditions(segment,state)
    conditions = state.conditions

    # default initial time, position, and mass
    t_initial = 0.0
    r_initial = conditions.frames.inertial.position_vector[0,:][None,:]
    m_initial = segment.analyses.weights.vehicle.mass_properties.takeoff

    # apply initials
    conditions = state.conditions
    conditions.weights.total_mass[:,0]   = m_initial
    conditions.frames.inertial.time[:,0] = t_initial
    conditions.frames.inertial.position_vector[:,:] = r_initial[:,:]

    throttle = segment.throttle	
    conditions.propulsion.throttle[:,0] = throttle



# ------------------------------------------------------------------
#   Methods For Post-Solver
# ------------------------------------------------------------------    

def post_process(segment,state):
    """ Segment.post_process(conditions,numerics,unknowns)
        post processes the conditions after converging the segment solver.
        Packs up the estimated distance for rotation in addition to the final 
        position vector found in the superclass post_process method.
    """

    conditions = Ground_Segment.post_process(self, conditions, numerics, unknowns)

    # process
    # Assume 3.5 seconds for rotation, with a constant groundspeed
    rotation_distance = conditions.frames.inertial.velocity_vector[-1,0] * 3.5

    # pack outputs
    conditions.frames.inertial.rotation_distance = np.ones([1,1])
    conditions.frames.inertial.rotation_distance[0,0] = rotation_distance
