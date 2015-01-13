
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from Constant_Speed_Constant_Altitude import Constant_Speed_Constant_Altitude

# import units
from SUAVE.Core import Units
km = Units.km
hr = Units.hr

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Constant_Mach_Constant_Altitude(Constant_Speed_Constant_Altitude):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  
    
    def __defaults__(self):
        self.tag = 'Constant Mach, Constant Altitude Cruise'
        
       # --- User Inputs
        
        self.altitude    = 10. * km
        self.mach        = 0.7
        self.distance    = 10. * km
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
        """ Segment.initialize_conditions(conditions)
            update the segment conditions
            pin down as many condition variables as possible in this function
            
            Assumptions:
                --
                
        """
        
        # unpack inputs
        alt  = self.altitude
        mach = self.mach
        atmo = self.analyses.atmosphere
        
        # compute speed, constant with constant altitude
        a = atmo.compute_values(alt,'speed_of_sound')
        self.air_speed = mach * a
        
        # call super class's initialize
        conditions = Constant_Speed_Constant_Altitude.initialize_conditions(self,conditions,numerics,initials)
        
        # done
        return conditions
    
    