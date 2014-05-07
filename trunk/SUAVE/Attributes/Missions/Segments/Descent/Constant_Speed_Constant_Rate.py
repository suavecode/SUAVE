
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from SUAVE.Attributes.Missions.Segments import Climb

# import units
from SUAVE.Attributes import Units
km = Units.km
hr = Units.hr
deg = Units.deg

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Constant_Speed_Constant_Rate(Climb.Constant_Speed_Constant_Rate):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  
    
    def __defaults__(self):
        self.tag = 'Constant Air Speed, Constant Rate Descent'
        
        # --- User Inputs
        
        self.altitude_start = None # Optional
        self.altitude_end   = 1. * km
        self.descent_rate   = 3.  * deg
        self.air_speed      = 100 * Units.m / Units.s
        
        return

    # ------------------------------------------------------------------
    #   Methods For Initialization
    # ------------------------------------------------------------------  
    
    def check_inputs(self):
        """ Segment.check():
            error checking of segment inputs
        """
        
        ## CODE
        
        return

    def initialize_conditions(self,conditions,numerics,initials=None):
        
        # unpack user inputs, set 'climb rate'
        descent_rate = self.descent_rate
        self.climb_rate = -descent_rate
        
        # everything else is the same
        Climb.Constant_Speed_Constant_Rate.initialize_conditions(self,conditions,numerics,initials)
        
        return conditions
        
