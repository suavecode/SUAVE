
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np
import time

# SUAVE imports
from SUAVE.Attributes.Missions.Segments import Aerodynamic_Segment

# import units
from SUAVE.Attributes import Units
km = Units.km
hr = Units.hr

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Constant_Speed_Constant_Altitude(Aerodynamic_Segment):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  
    
    def __defaults__(self):
        self.tag = 'Constant Speed, Constant Altitude Cruise'
        
       # --- User Inputs
        
        self.altitude  = None # Optional
        self.air_speed = 10. * km/hr
        self.distance  = 10. * km
        self.battery_energy = 0.0
        self.latitude       = 0.0
        self.longitude      = 0.0     

        # -- Conditions 
        
        # uses a ton of defaults from Aerodynamic_Segment
        
        
        # --- Unknowns
        unknowns = self.unknowns
        unknowns.controls.throttle = np.ones([1,1])*0.5
        unknowns.controls.theta    = np.zeros([1,1])
        
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
        """ Segment.initialize_conditions(conditions)
            update the segment conditions
            pin down as many condition variables as possible in this function
            
            Assumptions:
                --
                
        """
        
        # sets initial time and mass
        conditions = Aerodynamic_Segment.initialize_conditions(self,conditions,numerics,initials)
        
        # unpack inputs
        alt       = self.altitude
        xf        = self.distance
        air_speed = self.air_speed
        atmo      = self.atmosphere
        planet    = self.planet
        t_nondim  = numerics.dimensionless_time
        t_initial = conditions.frames.inertial.time[0,0]
        
        # check for initial altitude
        if alt is None:
            if not initials: raise AttributeError('altitude not set')
            alt = -1.0 * initials.frames.inertial.position_vector[0,2]
            self.altitude = alt
        
        # freestream details
        conditions.freestream.altitude[:,0] = alt
        conditions = self.compute_atmosphere(conditions,atmo)
        conditions = self.compute_gravity(conditions,planet)
        
        conditions.frames.inertial.velocity_vector[:,0] = air_speed
        conditions = self.compute_freestream(conditions)
        
        # dimensionalize time
        t_final = xf / air_speed + t_initial
        time =  t_nondim * (t_final-t_initial) + t_initial
        conditions.frames.inertial.time[:,0] = time[:,0]
        
        # positions (TODO: mamange user initials)
        conditions.frames.inertial.position_vector[:,2] = -alt # z points down
        
        # update differentials
        numerics = Aerodynamic_Segment.update_differentials(self,conditions,numerics)
        
        # done
        return conditions
    
    
    # ------------------------------------------------------------------
    #   Methods For Solver Iterations
    # ------------------------------------------------------------------    
    
    def update_differentials(self,conditions,numerics,unknowns):
        # time is constant for this segment,
        # don't need to update differential operators
        return numerics
    
    def update_conditions(self,conditions,numerics,unknowns):
        """ Segment.update_conditions(conditions,numerics,unknowns)
            if needed, updates the conditions given the current free unknowns and numerics
            called once per segment solver iteration
        """
        
        # unpack unknowns
        throttle = unknowns.controls.throttle
        theta    = unknowns.controls.theta
        
        # apply unknowns
        conditions.propulsion.throttle[:,0]            = throttle[:,0]
        conditions.frames.body.inertial_rotations[:,1] = theta[:,0]
        
        # angle of attack
        # external aerodynamics
        # propulsion
        # weights
        # total forces
        conditions = Aerodynamic_Segment.update_conditions(self,conditions,numerics,unknowns)
        
        return conditions
     
    def solve_residuals(self,conditions,numerics,unknowns,residuals):
        """ Segment.solve_residuals(conditions,numerics,unknowns,residuals)
            the hard work, solves the residuals for the free unknowns
            called once per segment solver iteration
        """
        
        # unpack inputs
        FT = conditions.frames.inertial.total_force_vector
        
        # process
        residuals.controls.Fx[:,0] = FT[:,0]
        residuals.controls.Fz[:,0] = FT[:,2]
        
        # pack outputs
        ## CODE
        
        return residuals
    
    # ------------------------------------------------------------------
    #   Methods For Post-Solver
    # ------------------------------------------------------------------    
    
    def post_process(self,conditions,numerics,unknowns):
        """ Segment.post_process(conditions,numerics,unknowns)
            post processes the conditions after converging the segment solver
            
            Usage Notes - 
                use this to store the unknowns and any other interesting in conditions
                    for later plotting
                for clarity and style, be sure to define any new fields in segment.__defaults__
            
        """
        
        x0 = conditions.frames.inertial.position_vector[0,0]
        vx = conditions.frames.inertial.velocity_vector[:,0]
        I  = numerics.integrate_time
        
        x = np.dot(I,vx) + x0
        
        conditions.frames.inertial.position_vector[:,0] = x
        
        conditions = Aerodynamic_Segment.post_process(self,conditions,numerics,unknowns)
        
        return conditions
    
    
        
    