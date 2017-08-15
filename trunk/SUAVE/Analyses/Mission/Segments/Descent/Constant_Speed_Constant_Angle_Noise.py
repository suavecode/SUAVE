## @ingroup Analyses-Mission-Segments-Descent
# Constant_Speed_Constant_Angle_Noise.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE

from SUAVE.Methods.Missions import Segments as Methods

from SUAVE.Analyses.Mission.Segments.Climb.Unknown_Throttle import Unknown_Throttle

# Units
from SUAVE.Core import Units


# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Descent
class Constant_Speed_Constant_Angle_Noise(Unknown_Throttle):
    """ Fixed at a true airspeed the vehicle will descend at a constant angle. This is a specific segment for Noise.
        A vehicle performs a descent to landing in accordance with a certification points for landing noise.
        Don't use this segment unless you're planning on post process for noise. It is slower than the normal constant
        speed constand angle segment.
    
        Assumptions:
        The points are linearly spaced rather than chebyshev spaced to give the proper "microphone" locations.
        
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
        self.altitude_start = None # Optional
        self.altitude_end   = 0.0 * Units.km
        self.descent_angle  = 3.  * Units.deg
        self.air_speed      = 100 * Units.m / Units.s
        self.state.numerics.discretization_method = SUAVE.Methods.Utilities.Chebyshev.linear_data
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
    
        initialize = self.process.initialize
        initialize.conditions   = Methods.Descent.Constant_Speed_Constant_Angle_Noise.initialize_conditions
        initialize.expand_state = Methods.Descent.Constant_Speed_Constant_Angle_Noise.expand_state
        
        return

