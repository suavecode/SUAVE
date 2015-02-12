
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from Unknown_Throttle import Unknown_Throttle

# import units
from SUAVE.Core import Units
km = Units.km
hr = Units.hr
deg = Units.deg

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Linear_Mach_Constant_Rate(Unknown_Throttle):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  
    
    def __defaults__(self):
        self.tag = 'Linear Mach, Constant Altitude Cruise'
        
        # --- User Inputs
        
        self.altitude_start = None # Optional
        self.altitude_end   = 10. * km
        self.climb_rate     = 3.  * Units.m / Units.s
        self.mach_number_end    = 0.7
        self.mach_number_start  = 0.8
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
        
        # sets altitude, atmospheric conditions,
        # initial time and mass
        # climb segments are discretized on altitude
        conditions = Unknown_Throttle.initialize_conditions(self,conditions,numerics,initials)

        # unpack user inputs
        climb_rate  = self.climb_rate
        
        # unpack conditions
        a = conditions.freestream.speed_of_sound
        
        Mo = self.mach_number_start
        Mf = self.mach_number_end
        
        mach_number = (Mf-Mo)*numerics.dimensionless_time + Mo
        
        # process
        v_mag = mach_number * a
        v_z   = -climb_rate # z points down
        v_x   = np.sqrt( v_mag**2 - v_z**2 )
        
        conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
        conditions.frames.inertial.velocity_vector[:,2] = v_z
        
        # freestream conditions
        # freestream.velocity, dynamic pressure, mach number, ReL
        conditions = self.compute_freestream(conditions)  
        
        return conditions
        
