
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses.Mission.Segments import Aerodynamic
from SUAVE.Analyses.Mission.Segments import Conditions

from SUAVE.Methods.Missions import Segments as Methods

from SUAVE.Analyses import Process
from SUAVE.Analyses.Mission.Segments.Climb.Unknown_Throttle import Unknown_Throttle

# Units
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

class Climb(Unknown_Throttle):
    
    def __defaults__(self):
        
        # --------------------------------------------------------------
        #   User inputs
        # --------------------------------------------------------------
        self.altitude_start = None # Optional
        self.altitude_end   = 1. * Units.km
        self.climb_rate     = 1.  * Units.m / Units.s
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        initialize = self.process.initialize
        initialize.conditions = Methods.Hover.Climb.initialize_conditions
    
        return
       