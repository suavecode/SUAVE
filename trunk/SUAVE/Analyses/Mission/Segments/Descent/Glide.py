## @ingroup Analyses-Mission-Segments-Climb
# Optimized.py
#
# Created:  Mar 2016, E. Botero 
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses.Mission.Segments import Aerodynamic
from SUAVE.Analyses.Mission.Segments import Conditions
from SUAVE.Analyses.Mission.Segments.Climb import Optimized

from SUAVE.Methods.Missions import Segments as Methods

from SUAVE.Analyses import Process

# Units
from SUAVE.Core import Units
import SUAVE

import numpy as np

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Climb
class Glide(Optimized):
    """ Optimize your climb segment. This is useful if you're not sure how your vehicle should climb.
        You can set any conditions parameter as the objective, for example setting time to climb or vehicle mass:
        segment.objective       = 'conditions.weights.total_mass[-1,0]'
        
        The ending airspeed is an optional parameter.
        
        This segment takes far longer to run than a normal segment. Wrapping this into a vehicle optimization
        has not yet been tested for robustness.
        
    
        Assumptions:
        Can use SNOPT if you have it installed through PyOpt. But defaults to SLSQP through 
        Runs a linear true airspeed mission first to initialize conditions.
        
        Source:
        None
    """          
    
    def __defaults__(self):
        """ This sets the default solver flow. Anything in here can be modified after initializing a segment.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """          
        
        # --------------------------------------------------------------
        #   User inputs
        # --------------------------------------------------------------
        self.altitude_start  = None
        self.altitude_end    = None
        self.air_speed_start = None
        self.air_speed_end   = None
        self.objective       = 'conditions.frames.inertial.position_vector[-1,0]'
        self.minimize        = False
        self.CL_limit        = 1.e20 
        self.seed_climb_rate = -100. * Units['feet/min']
        self.algorithm       = 'SLSQP'
        
        # initials and unknowns
        ones_row    = self.state.ones_row
        self.state.unknowns.__delitem__('throttle')
        
        initialize = self.process.initialize
        initialize.solved_mission          = solve_linear_speed_constant_rate

        iterate = self.process.iterate
        iterate.unknowns.mission           = unpack_unknowns
        iterate.conditions.propulsion      = update_thrust

        return

def unpack_unknowns(segment,state):
    
    """Unpacks the unknowns set in the mission to be available for the mission.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    state.unknowns.throttle            [Unitless]
    state.unknowns.body_angle          [Radians]
    state.unknowns.flight_path_angle   [Radians]
    state.unknowns.velocity            [meters/second]
    segment.altitude_start             [meters]
    segment.altitude_end               [meters]
    segment.air_speed_start            [meters/second]
    segment.air_speed_end              [meters/second]

    Outputs:
    state.conditions.propulsion.throttle            [Unitless]
    state.conditions.frames.body.inertial_rotations [Radians]
    conditions.frames.inertial.velocity_vector      [meters/second]

    Properties Used:
    N/A
    """    
    
    # unpack unknowns and givens
    theta    = state.unknowns.body_angle
    gamma    = state.unknowns.flight_path_angle
    vel      = state.unknowns.velocity
    alt0     = segment.altitude_start
    altf     = segment.altitude_end
    vel0     = segment.air_speed_start
    velf     = segment.air_speed_end 

    # Overide the speeds   
    if segment.air_speed_end is None:
        v_mag =  np.concatenate([[[vel0]],vel])
    elif segment.air_speed_end is not None:
        v_mag = np.concatenate([[[vel0]],vel,[[velf]]])
        
    if np.all(gamma == 0.):
        gamma[gamma==0.] = 1.e-16
        
    if np.all(vel == 0.):
        vel[vel==0.] = 1.e-16
    
    # process velocity vector
    v_x   =  v_mag * np.cos(gamma)
    v_z   = -v_mag * np.sin(gamma)    

    # apply unknowns and pack conditions   
    state.conditions.propulsion.throttle[:,0]             = np.zeros_like(v_x[:,0])
    state.conditions.frames.body.inertial_rotations[:,1]  = theta[:,0]   
    state.conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0] 
    state.conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0] 
    
# ----------------------------------------------------------------------
#  Update Thrust
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def update_thrust(segment,state):
    """ Evaluates the energy network to find the thrust force and mass rate

        Inputs -
            segment.analyses.energy_network    [Function]
            state                              [Data]

        Outputs -
            state.conditions:
               frames.body.thrust_force_vector [Newtons]
               weights.vehicle_mass_rate       [kg/s]


        Assumptions -


    """    
    ones_row = state.ones_row

    # pack conditions
    conditions = state.conditions
    conditions.frames.body.thrust_force_vector = 0. * ones_row(3)
    conditions.weights.vehicle_mass_rate       = 0. * ones_row(1)
    
    

## @ingroup Methods-Missions-Segments-Climb
def solve_linear_speed_constant_rate(segment,state):
    
    """ The sets up an solves a mini segment that is a linear speed constant rate segment. The results become the initial conditions for an optimized climb segment later

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    segment.altitude_start             [meters]
    segment.altitude_end               [meters]
    segment.air_speed_start            [meters/second]
    segment.air_speed_end              [meters/second]
    segment.analyses                   [Data]
    state.numerics                     [Data]

    Outputs:
    state.unknowns.throttle            [Unitless]
    state.unknowns.body_angle          [Radians]
    state.unknowns.flight_path_angle   [Radians]
    state.unknowns.velocity            [meters/second]
    
    Properties Used:
    N/A    
    
    """
    
    mini_mission = SUAVE.Analyses.Mission.Sequential_Segments()
    
    LSCR = SUAVE.Analyses.Mission.Segments.Climb.Linear_Speed_Constant_Rate()
    LSCR.air_speed_start = segment.air_speed_start
    
    if segment.air_speed_end is not None:
        LSCR.air_speed_end   = segment.air_speed_end
    else:
        LSCR.air_speed_end   = segment.air_speed_start
        
    LSCR.altitude_start   = segment.altitude_start
    LSCR.altitude_end     = segment.altitude_end
    LSCR.climb_rate       = segment.seed_climb_rate
    LSCR.analyses         = segment.analyses
    LSCR.state.conditions = state.conditions
    LSCR.state.numerics   = state.numerics
    mini_mission.append_segment(LSCR)
    
    results = mini_mission.evaluate()
    LSCR_res = results.segments.analysis
    
    state.unknowns.body_angle        = LSCR_res.unknowns.body_angle
    state.unknowns.flight_path_angle = LSCR_res.unknowns.body_angle - LSCR_res.conditions.aerodynamics.angle_of_attack
    
    # Make the velocity vector
    v_mag = np.linalg.norm(LSCR_res.conditions.frames.inertial.velocity_vector,axis=1)
    
    if segment.air_speed_end is None:
        state.unknowns.velocity =  np.reshape(v_mag[1:],(-1, 1))
    elif segment.air_speed_end is not None:    
        state.unknowns.velocity = np.reshape(v_mag[1:-1],(-1, 1))    