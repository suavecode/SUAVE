# Landing.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------


# SUAVE imports
from Ground import Ground
from SUAVE.Methods.Missions import Segments as Methods

# Units
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Landing(Ground):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  

    def __defaults__(self):

        self.velocity_start       = 150 * Units.knots
        self.velocity_end         = 0.0
        self.friction_coefficient = 0.4
        self.throttle             = 0.0
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
    
        initialize = self.process.initialize
        initialize.conditions = Methods.Ground.Landing.initialize_conditions

        return