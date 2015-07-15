
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses.Mission.Segments import Aerodynamic
from SUAVE.Analyses.Mission.Segments import Conditions

from SUAVE.Methods.Missions import Segments as Methods

from SUAVE.Analyses import Process
from Unknown_Throttle import Unknown_Throttle

# Units
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

class Constant_Speed_Constant_Rate(Unknown_Throttle):
    
    def __defaults__(self):
        
        # --------------------------------------------------------------
        #   User inputs
        # --------------------------------------------------------------
        self.altitude_start = None # Optional
        self.altitude_end   = 10. * Units.km
        self.climb_rate     = 3.  * Units.m / Units.s
        self.air_speed      = 100 * Units.m / Units.s
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        initialize = self.process.initialize
        initialize.conditions = Methods.Climb.Constant_Speed_Constant_Rate.initialize_conditions
    
        return
       