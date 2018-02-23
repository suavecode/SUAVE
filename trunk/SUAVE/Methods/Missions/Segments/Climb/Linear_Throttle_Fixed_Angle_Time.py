## @ingroup Methods-Missions-Segments-Climb
# Constant_Speed_Constant_Angle.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Climb
def initialize_conditions(segment,state):
        
    # Setup the time
    
    # Setup the initial altitudes and speeds
    
    # Set the throttle
    
    pass
    
    
    
# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Climb
def unpack_unknowns(segment,state):
    """Unpacks the unknowns set in the mission to be available for the mission.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    state.unknowns.throttle            [Unitless]
    state.unknowns.body_angle          [Radians]

    Outputs:
    state.conditions.propulsion.throttle            [Unitless]
    state.conditions.frames.body.inertial_rotations [Radians]

    Properties Used:
    N/A
    """        
    
    # unpack unknowns
    self.state.unknowns.body_angle = ones_row(1) * 1.0 * Units.degrees
    self.state.unknowns.velocity   = ones_row(1) * 1.0 * Units.m / Units.s
    

    # integrate velocities to get altitudes
    
    #
    
    # apply unknowns
