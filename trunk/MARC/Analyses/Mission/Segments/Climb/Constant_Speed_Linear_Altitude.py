## @ingroup Analyses-Mission-Segments-Climb
# Constant_Speed_Linear_Altitude.py
#
# Created:  
# Modified: Jun 2017, E. Botero
#           Apr 2020, M. Clarke
#           Aug 2021, R. Erhard
#           Nov 2022, M. Clarke

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
class Constant_Speed_Linear_Altitude(Unknown_Throttle):
    """ Climb at a constant true airspeed but linearly change altitudes over a distance.
    
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
        self.air_speed       = 10. * Units['km/hr']
        self.distance        = 10. * Units.km
        self.altitude_start  = None
        self.altitude_end    = None
        self.true_course     = 0.0 * Units.degrees    
        
 
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        initialize = self.process.initialize
        
        initialize.conditions = Methods.Climb.Constant_Speed_Linear_Altitude.initialize_conditions  

        return