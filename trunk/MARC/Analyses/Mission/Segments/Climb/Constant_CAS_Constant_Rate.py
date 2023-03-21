## @ingroup Analyses-Mission-Segments-Climb
# Constant_CAS_Constant_Rate.py
#
# Created:  Nov 2020, S. Karpuk
# Modified: Aug 2021, J. Mukhopadhaya
#
# Adapted from Constant_CAS_Constant_Rate

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
class Constant_CAS_Constant_Rate(Unknown_Throttle):
    """ Climb at a constant Calibrated Airspeed (CAS) at a constant rate.
    
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
        self.altitude_start       = None # Optional
        self.altitude_end         = 10. * Units.km
        self.climb_rate           = 3.  * Units.m / Units.s
        self.calibrated_air_speed = None
        self.true_course_angle    = 0.0 * Units.degrees
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        initialize = self.process.initialize
        
        initialize.conditions = Methods.Climb.Constant_CAS_Constant_Rate.initialize_conditions
    
        self.process.initialize = initialize
        
        return
       
