
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from Climb_Segment import Climb_Segment

# import units
from SUAVE.Attributes import Units
km = Units.km
hr = Units.hr
deg = Units.deg

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Constant_Throttle_Constant_Speed(Climb_Segment):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  
    
    def __defaults__(self):
        self.tag = 'Constant Mach, Constant Altitude Cruise'
        
        # --- User Inputs
        
        self.altitude_start = None # Optional
        self.altitude_end   = 10. * km
        self.throttle       = 0.5
        self.air_speed      = 100 * Units.m / Units.s
        
        # -- Conditions 
        
        # uses a ton of defaults from Aerodynamic_Segment
        
        
        # --- Unknowns
        unknowns = self.unknowns
        unknowns.controls.body_angle = np.zeros([1,1]) # aircraft inertial to body angle
        unknowns.controls.wind_angle = np.zeros([1,1]) # aircraft inertial to wind angle
        
        # --- Residuals
        residuals = self.residuals
        residuals.controls.Fx = np.ones([1,1])
        residuals.controls.Fz = np.ones([1,1])
        
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
        conditions = Climb_Segment.initialize_conditions(self,conditions,numerics,initials)
        
        # unpack user inputs
        throttle   = self.throttle
        air_speed  = self.air_speed
        
        # process freestream
        # gets freestream.velocity_mag, mach number, dynamic pressure, ReL
        conditions.frames.inertial.velocity_vector[:,0] = air_speed # start up value
        conditions = self.compute_freestream(conditions)
        
        # pack conditions
        conditions.propulsion.throttle[:,0] = throttle
        
        return conditions
        
        

    # ------------------------------------------------------------------
    #   Methods For Iterations
    # ------------------------------------------------------------------  
    
    def update_velocity_vector(self,unknowns,conditions):
        """ helper function to set velocity_vector
            called from update_differentials()
        """
        
        # unpack
        v_mag = self.air_speed
        theta = unknowns.controls.wind_angle[:,0][:,None]
        
        # process
        v_x = v_mag * np.cos(theta)
        v_z = -v_mag * np.sin(theta) # z points down
        
        # pack
        conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
        conditions.frames.inertial.velocity_vector[:,2] = v_z[:,0]
        
        return conditions
    
    def update_conditions(self,conditions,numerics,unknowns):
        """ Segment.update_conditions(conditions,numerics,unknowns)
            
        """
        
        # unpack/pack unknowns
        theta = unknowns.controls.body_angle
        conditions.frames.body.inertial_rotations[:,1] = theta[:,0]
        
        # unpack conditions
        v = conditions.frames.inertial.velocity_vector
        D = numerics.differentiate_time
        
        # accelerations
        acc = np.dot(D,v)
        
        # pack conditions
        conditions.frames.inertial.acceleration_vector[:,:] = acc[:,:]
        
        # angle of attack
        # external aerodynamics
        # propulsion
        # weights
        # total forces
        conditions = Climb_Segment.update_conditions(self,conditions,numerics,unknowns)
        
        return conditions
    
    def solve_residuals(self,conditions,numerics,unknowns,residuals):
        """ Segment.solve_residuals(conditions,numerics,unknowns,residuals)
            the hard work, solves the residuals for the free unknowns
            called once per segment solver iteration
        """
        
        # unpack inputs
        FT = conditions.frames.inertial.total_force_vector
        a  = conditions.frames.inertial.acceleration_vector
        
        # process
        residuals.controls.Fx[:,0] = FT[:,0] - a[:,0]
        residuals.controls.Fz[:,0] = FT[:,2] - a[:,2]
        
        return residuals