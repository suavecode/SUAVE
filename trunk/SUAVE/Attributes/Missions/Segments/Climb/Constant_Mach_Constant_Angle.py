
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from Unknown_Throttle import Unknown_Throttle

# import units
from SUAVE.Attributes import Units
km = Units.km
hr = Units.hr
deg = Units.deg

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Constant_Mach_Constant_Angle(Unknown_Throttle):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  
    
    def __defaults__(self):
        self.tag = 'Constant Mach, Constant Altitude Cruise'
        
        # --- User Inputs
        
        self.altitude_start = 1.  * km
        self.altitude_end   = 10. * km
        self.climb_angle    = 3.  * deg
        self.mach_number    = 0.7 
        
                
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
        
        # sets altitude, atmospheric conditions,
        # initial time and mass
        # climb segments are discretized on altitude
        conditions = Unknown_Throttle.initialize_conditions(self,conditions,numerics,initials)

        # unpack user inputs
        climb_angle = self.climb_angle
        mach_number = self.mach_number
        
        # unpack conditions
        a = conditions.freestream.speed_of_sound
        
        # velocity vector
        v_mag = mach_number * a
        v_x   = v_mag * np.cos(climb_angle)
        v_z   = -v_mag * np.sin(climb_angle)
        
        conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
        conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0]
        
        # freestream conditions
        # freestream.velocity, dynamic pressure, mach number, ReL
        conditions = self.compute_freestream(conditions)
        
        return conditions
        
