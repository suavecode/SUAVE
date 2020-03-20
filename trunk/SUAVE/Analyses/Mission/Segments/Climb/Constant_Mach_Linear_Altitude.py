## @ingroup Analyses-Mission-Segments-Climb
# Constant_Mach_Linear_Altitude.py
#
# Created:  June 2017, E. Botero
# Modified: Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from .Constant_Speed_Linear_Altitude import Constant_Speed_Linear_Altitude

from SUAVE.Methods.Missions import Segments as Methods

# Units
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Climb
class Constant_Mach_Linear_Altitude(Constant_Speed_Linear_Altitude):
    """ Climb at a constant mach number but linearly change altitudes over a distance.
    
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
        self.altitude  = None
        self.mach      = 0.5
        self.distance  = 10. * Units.km
        self.altitude_start = None
        self.altitude_end = None
        
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # only need to change one setup step from constant_speed_constant_altitude
        initialize = self.process.initialize
        initialize.conditions = Methods.Climb.Constant_Mach_Linear_Altitude.initialize_conditions
        

        return

