import numpy as np
from SUAVE.Methods.Geometry.Three_Dimensional \
     import angles_to_dcms, orientation_product, orientation_transpose
import SUAVE

def unpack_unknowns(segment,state):
    
    # unpack unknowns
    unknowns   = state.unknowns
    velocity_x = unknowns.velocity_x
    final_time = unknowns.time

    #apply unknowns
    conditions = state.conditions
    conditions.frames.inertial.velocity_vector[:,0] = velocity_x[:,0]
    conditions.frames.inertial.velocity_vector[velocity_x[:,0]==0.0,0] = 0.1
    
    return

def initialize_conditions(segment,state):
    """ Segment.initialize_conditions(conditions,numerics,initials=None)
        update the segment conditions
        pin down as many condition variables as possible in this function
        Inputs:
            conditions - the conditions data dictionary, with initialized
            zero arrays, with number of rows = 
            segment.conditions.n_control_points
            initials - a data dictionary with 1-row column arrays pulled from
            the last row of the previous segment's conditions data, or none
            if no previous segment
        Outputs:
            conditions - the conditions data dictionary, updated with the 
                         values that can be precalculated
        Assumptions:
            --
        Usage Notes:
            may need to inspect segment (self) for user inputs
            will be called before solving the segments free unknowns
    """

    conditions = state.conditions

    # unpack inputs
    v0       = segment.velocity_start
    vf       = segment.velocity_end
    conditions.ground.incline[:,0]              = segment.ground_incline
    conditions.ground.friction_coefficient[:,0] = segment.friction_coefficient
    N        = len(conditions.frames.inertial.velocity_vector[:,0])

    # avoid having zero velocity since aero and propulsion models need non-zero Reynolds number
    if v0 == 0.0: v0 = 0.01
    if vf == 0.0: vf = 0.01

    # repack
    segment.velocity_start = v0
    segment.velocity_end   = vf

    # pack conditions
    conditions.frames.inertial.velocity_vector[:,0] = np.linspace(v0,vf,N)


def compute_ground_forces(segment,state):
    """ Compute the rolling friction on the aircraft """

    # unpack
    conditions             = state.conditions
    W                      = conditions.frames.inertial.gravity_force_vector[:,2,None]
    friction_coeff         = conditions.ground.friction_coefficient
    wind_lift_force_vector = conditions.frames.wind.lift_force_vector

    #transformation matrix to get lift in inertial frame
    T_wind2inertial = conditions.frames.wind.transform_to_inertial

    # to inertial frame
    L = orientation_product(T_wind2inertial,wind_lift_force_vector)[:,2,None]

    #compute friction force
    N  = -(W + L)
    Ff = N * friction_coeff

    #pack results. Friction acts along x-direction
    conditions.frames.inertial.ground_force_vector[:,2] = N[:,0]
    conditions.frames.inertial.ground_force_vector[:,0] = Ff[:,0]



def compute_forces(segment,state):


    SUAVE.Methods.Missions.Segments.Common.Frames.update_forces(segment,state)

    # unpack forces
    conditions                   = state.conditions
    total_aero_forces            = conditions.frames.inertial.total_force_vector
    inertial_ground_force_vector = conditions.frames.inertial.ground_force_vector

    # sum of the forces, including friction force
    F = total_aero_forces + inertial_ground_force_vector

    # pack
    conditions.frames.inertial.total_force_vector[:,:] = F[:,:]



def solve_residuals(segment,state):
    """ Segment.solve_residuals(conditions,numerics,unknowns,residuals)
        the hard work, solves the residuals for the free unknowns
        called once per segment solver iteration
    """

    # unpack inputs
    conditions = state.conditions
    FT = conditions.frames.inertial.total_force_vector
    vf = segment.velocity_end
    v  = conditions.frames.inertial.velocity_vector
    D  = state.numerics.dimensionless.differentiate
    m  = conditions.weights.total_mass

    # process and pack
    acceleration = np.dot(D , v)
    conditions.frames.inertial.acceleration_vector = acceleration

    state.residuals.final_velocity_error = (v[-1,0] - vf)
    state.residuals.acceleration_x       = (FT / m - acceleration)


# ------------------------------------------------------------------
#   Methods For Post-Solver
# ------------------------------------------------------------------    

def post_process(segment,state):
    """ Segment.post_process(conditions,numerics,unknowns)
        post processes the conditions after converging the segment solver.
        Packs up the final position vector to allow estimation of the ground
        roll distance (e.g., distance from brake release to rotation speed in
        takeoff, or distance from touchdown to full stop on landing).
        Inputs - 
            unknowns - data dictionary of converged segment free unknowns with
            fields:
                states, controls, finals
                    these are defined in segment.__defaults__
            conditions - data dictionary of segment conditions
                    these are defined in segment.__defaults__
            numerics - data dictionary of the converged differential operators
        Outputs - 
            conditions - data dictionary with remaining fields filled with post-
            processed conditions. Updated fields are:
            conditions.frames.inertial.position_vector  (x-position update)
        Usage Notes - 
            Use this to store the unknowns and any other interesting in 
            conditions for later plotting. For clarity and style, be sure to 
            define any new fields in segment.__defaults__
    """

    # unpack inputs
    ground_velocity  = conditions.frames.inertial.velocity_vector
    I                = numerics.integrate_time
    initial_position = conditions.frames.inertial.position_vector[0,:]

    # process
    position_vector = initial_position + np.dot( I , ground_velocity)

    # pack outputs
    conditions.frames.inertial.position_vector = position_vector
