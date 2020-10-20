## @ingroup Analyses-Mission-Segments-Descent
# Constant_EAS_Constant_Rate.py
#
# Created:  Aug 2016, T. MacDonald
# Modified: 
#
# Adapted from Constant_Speed_Constant_Rate

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Methods.Missions import Segments as Methods

from .Unknown_Throttle import Unknown_Throttle

# Units
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Descent
class Constant_EAS_Constant_Rate(Unknown_Throttle):
    """ Fixed at an Equivalent Airspeed (EAS) the vehicle will descent at a constant rate.
    
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
        self.descent_rate         = 3.  * Units.m / Units.s
        self.equivalent_air_speed = 100 * Units.m / Units.s
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        initialize = self.process.initialize
        
        initialize.conditions = Methods.Descent.Constant_EAS_Constant_Rate.initialize_conditions
    
        self.process.initialize = initialize
        
        return
       