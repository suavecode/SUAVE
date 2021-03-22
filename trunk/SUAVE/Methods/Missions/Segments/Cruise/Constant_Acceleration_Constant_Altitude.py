## @ingroup Methods-Missions-Segments-Cruise
# Constant_Acceleration_Constant_Altitude.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero
#           May 2019, T. MacDonald 
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Cruise
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Constant acceleration and constant altitude

    Source:
    N/A

    Inputs:
    segment.altitude                [meters]
    segment.air_speed_start         [meters/second]
    segment.air_speed_end           [meters/second]
    segment.acceleration            [meters/second^2]
    conditions.frames.inertial.time [seconds]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]
    conditions.frames.inertial.time             [seconds]

    Properties Used:
    N/A
    """      
    
    # unpack
    alt = segment.altitude 
    v0  = segment.air_speed_start
    vf  = segment.air_speed_end
    ax  = segment.acceleration   
    conditions = segment.state.conditions 
    
    # check for initial altitude
    if alt is None:
        if not segment.state.initials: raise AttributeError('altitude not set')
        alt = -1.0 * segment.state.initials.conditions.frames.inertial.position_vector[-1,2]
    
    # dimensionalize time
    t_initial = conditions.frames.inertial.time[0,0]
    t_final   = (vf-v0)/ax + t_initial
    t_nondim  = segment.state.numerics.dimensionless.control_points
    time      = t_nondim * (t_final-t_initial) + t_initial
    
    # Figure out vx
    vx = v0+(time - t_initial)*ax
    
    # pack
    segment.state.conditions.freestream.altitude[:,0] = alt
    segment.state.conditions.frames.inertial.position_vector[:,2] = -alt # z points down
    segment.state.conditions.frames.inertial.velocity_vector[:,0] = vx[:,0]
    segment.state.conditions.frames.inertial.time[:,0] = time[:,0]
    

# ----------------------------------------------------------------------
#  Residual Total Forces
# ----------------------------------------------------------------------
    
## @ingroup Methods-Missions-Segments-Cruise    
def residual_total_forces(segment):
    """ Calculates a residual based on forces
    
        Assumptions:
        The vehicle is not accelerating, doesn't use gravity
        
        Inputs:
            segment.acceleration                   [meters/second^2]
            segment.state.ones_row                 [vector]
            state.conditions:
                frames.inertial.total_force_vector [Newtons]
                weights.total_mass                 [kg]
            
        Outputs:
            state.conditions:
                state.residuals.forces [meters/second^2]

        Properties Used:
        N/A
                                
    """      
    
    # Unpack
    FT      = segment.state.conditions.frames.inertial.total_force_vector
    ax      = segment.acceleration 
    m       = segment.state.conditions.weights.total_mass  
    one_row = segment.state.ones_row
    
    a_x    = ax*one_row(1)
    
    # horizontal
    segment.state.residuals.forces[:,0] = FT[:,0]/m[:,0] - a_x[:,0]
    # vertical
    segment.state.residuals.forces[:,1] = FT[:,2]/m[:,0] 

    return
