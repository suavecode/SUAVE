## @ingroup Analyses-Mission-Segments-Climb
# Linear_Speed_Constant_Rate.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# MARC imports
from MARC.Methods.Missions import Segments as Methods

from .Unknown_Throttle import Unknown_Throttle

# Units
from MARC.Core import Units


# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Climb
class Linear_Speed_Constant_Rate(Unknown_Throttle):
    """ Linearly change true airspeed while climbing at a constant rate.
    
        Assumptions:
        None
        
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
        self.altitude_start    = None # Optional
        self.altitude_end      = 10. * Units.km
        self.climb_rate        = 3.  * Units.m / Units.s
        self.air_speed_start   = None
        self.air_speed_end     = 200 * Units.m / Units.s
        self.true_course_angle = 0.0 * Units.degrees    
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        initialize = self.process.initialize
        initialize.conditions = Methods.Climb.Linear_Speed_Constant_Rate.initialize_conditions 

        return

