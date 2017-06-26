# Constant_Mach_Linear_Altitude.py
#
# Created:  June 2017, E. Botero
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses.Mission.Segments.Cruise import Constant_Speed_Linear_Altitude

from SUAVE.Methods.Missions import Segments as Methods

# Units
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

class Constant_Mach_Linear_Altitude(Constant_Speed_Linear_Altitude):
    
    def __defaults__(self):
        
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
        initialize.conditions = Methods.Cruise.Constant_Mach_Linear_Altitude.initialize_conditions
        

        return

