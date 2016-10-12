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

from Unknown_Throttle import Unknown_Throttle

# Units
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

class Constant_EAS_Constant_Rate(Unknown_Throttle):
    
    def __defaults__(self):
        
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
       