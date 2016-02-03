# Constant_Dynamic_Pressure_Constant_Rate.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Methods.Missions import Segments as Methods

from Unknown_Throttle import Unknown_Throttle

# Units
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

class Constant_Dynamic_Pressure_Constant_Rate(Unknown_Throttle):
    
    def __defaults__(self):
        
        # --------------------------------------------------------------
        #   User inputs
        # --------------------------------------------------------------
        self.altitude_start   = None # Optional
        self.altitude_end     = 10. * Units.km
        self.climb_rate       = 3.  * Units.m / Units.s
        self.dynamic_pressure = 1600 * Units.pascals
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        initialize = self.process.initialize
        initialize.conditions = Methods.Climb.Constant_Dynamic_Pressure_Constant_Rate.initialize_conditions
    
        return
       