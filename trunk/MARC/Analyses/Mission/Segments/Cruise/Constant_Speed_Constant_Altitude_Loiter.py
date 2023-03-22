## @ingroup Analyses-Mission-Segments-Cruise
# Constant_Speed_Constant_Altitude_Loiter.py
#
# Created:  Jun 2017, E. Botero 
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# MARC imports
from MARC.Methods.Missions import Segments as Methods

from .Constant_Speed_Constant_Altitude import Constant_Speed_Constant_Altitude

# Units
from MARC.Core import Units

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Cruise
class Constant_Speed_Constant_Altitude_Loiter(Constant_Speed_Constant_Altitude):
    """ Fixed true airspeed and altitude for a fixed time.
        This is useful aircraft who need to station keep.
    
        Assumptions:
        Built off of a constant speed constant altitude segment
        
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
        self.altitude          = None
        self.air_speed         = None
        self.time              = 1.0 * Units.sec
        self.true_course_angle = 0.0 * Units.degrees  
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        initialize = self.process.initialize
        initialize.conditions = Methods.Cruise.Constant_Speed_Constant_Altitude_Loiter.initialize_conditions


        return

