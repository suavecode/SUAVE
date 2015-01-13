
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np
import time

# SUAVE imports
from SUAVE.Analyses.Missions.Segments import Climb

# import units
from SUAVE.Core import Units
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
        self.battery_energy = 0.0
        self.latitude       = 0.0
        self.longitude      = 0.0        
    
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
        
