
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np
import time

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

class Unknown_Throttle(Climb_Segment):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  
    
    def __defaults__(self):
        self.tag = 'Constant Mach, Constant Altitude Cruise'
        
        # --- User Inputs
        
        self.altitude_start = None # Optional
        self.altitude_end   = 10. * km
        self.battery_energy = 0.0
        self.latitude       = 0.0
        self.longitude      = 0.0       
   
        # -- Conditions 
        
        # uses a ton of defaults from Aerodynamic_Segment
        
        
        # --- Unknowns
        unknowns = self.unknowns
        unknowns.controls.throttle = np.ones([1,1])*0.5 # engine throttle
        unknowns.controls.theta    = np.zeros([1,1]) # aircraft inertial to body angle
        
        # --- Residuals
        residuals = self.residuals
        residuals.controls.Fx = np.ones([1,1])
        residuals.controls.Fz = np.ones([1,1])
        
        return
        

    # ------------------------------------------------------------------
    #   Methods For Iterations
    # ------------------------------------------------------------------  
    
    def update_conditions(self,conditions,numerics,unknowns):
        """ Segment.update_conditions(conditions,numerics,unknowns)
            
        """
        
        # unpack unknowns
        throttle = unknowns.controls.throttle
        theta    = unknowns.controls.theta # body angle
        
        # apply unknowns
        conditions.propulsion.throttle[:,0]            = throttle[:,0]
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
            solves the residuals for the free unknowns
            called once per segment solver iteration
        """
        
        # unpack inputs
        FT = conditions.frames.inertial.total_force_vector
        a  = conditions.frames.inertial.acceleration_vector
        m  = conditions.weights.total_mass
        
        # process
        residuals.controls.Fx[:,0] = FT[:,0]/m[:,0] - a[:,0]
        residuals.controls.Fz[:,0] = FT[:,2]/m[:,0] - a[:,2]
        
        return residuals