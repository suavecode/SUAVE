
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from SUAVE.Attributes.Missions.Segments import Aerodynamic_Segment

# import units
from SUAVE.Attributes import Units
km = Units.km
hr = Units.hr

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Constant_Speed_Altitude_Radius_Loiter(Aerodynamic_Segment):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  
    
    def __defaults__(self):
        self.tag = 'Constant Speed, Constant Altitude Constant Radius Loiter '
        
       # --- User Inputs
        
        self.altitude  = None # Optional
        self.air_speed = 10. * km/hr
        self.distance  = 10. * km

        
        # -- Conditions 
        
        # uses a ton of defaults from Aerodynamic_Segment
        
        
        # --- Unknowns
        unknowns = self.unknowns
        unknowns.controls.throttle = np.ones([1,1])
        unknowns.controls.theta    = np.zeros([1,1])
        unknowns.controls.psi      = np.zeros([1,1])
        
        # --- Residuals
        residuals = self.residuals
        residuals.controls.Fx = np.ones([1,1])
        residuals.controls.Fy = np.ones([1,1])
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
        seg_time  = self.time
        air_speed = self.air_speed
        atmo      = self.atmosphere
        planet    = self.planet
        radius    = self.radius
        t_nondim  = numerics.dimensionless_time
        t_initial = conditions.frames.inertial.time[0,0]
        
        # check for initial altitude
        if alt is None:
            if not initials: raise AttributeError('altitude not set')
            alt = -1.0 * initials.frames.inertial.position_vector[0,2]
            self.altitude = alt
        
        # calculate bank angle
        conditions = self.compute_gravity(conditions,planet)  
        ar    = (air_speed**2)/radius # Centripetal acceleration
        g     = conditions.freestream.gravity[:,0] 
        omega = air_speed/radius
        phi   = np.arctan(ar/g)
        
        conditions.frames.body.inertial_rotations[:,0] = phi
        
        # dimensionalize time
        time =  t_nondim * (seg_time) + t_initial
        conditions.frames.inertial.time[:,0] = time[:,0]
        
        angle    = np.remainder(time*omega,2*np.pi)
        
        # the acceleration in the inertial frame
        conditions.frames.inertial.acceleration_vector[:,0] = np.sin(angle[:,0])*ar
        conditions.frames.inertial.acceleration_vector[:,1] = np.cos(angle[:,0])*ar
        
        # the velocity in the inertial frame
        conditions.frames.inertial.velocity_vector[:,0] = air_speed*np.cos(angle[:,0])        
        conditions.frames.inertial.velocity_vector[:,1] = air_speed*np.sin(angle[:,0])

        # freestream details
        conditions.freestream.altitude[:,0] = alt
        conditions = self.compute_atmosphere(conditions,atmo)      
        conditions = self.compute_freestream(conditions)
        
        # this is psi
        conditions.frames.body.inertial_rotations[:,2] = angle[:,0]
        
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
        """ Segment.update_differentials(conditions, numerics, unknowns)
            updates the differential operators t, D and I
            must return in dimensional time, with t[0] = 0
            
            Works with a segment discretized in vertical position, altitude
            
            Inputs - 
                unknowns      - data dictionary of segment free unknowns
                conditions    - data dictionary of segment conditions
                numerics - data dictionary of non-dimensional differential operators
                
            Outputs - 
                numerics - updated data dictionary with dimensional numerics 
            
            Assumptions - 
                outputed operators are in dimensional time for the current solver iteration
                works with a segment discretized in vertical position, altitude
                
        """
        
        # unpack
        t = numerics.dimensionless_time
        D = numerics.differentiate_dimensionless
        I = numerics.integrate_dimensionless
        
        times = conditions.frames.inertial.time[:,0]
        
        # get overall time step
        dt = times[-1]
        
        # rescale operators
        D = D / dt
        I = I * dt
        t = t * dt
    
        # pack
        numerics.time = t
        numerics.differentiate_time = D
        numerics.integrate_time = I      
        
        return numerics
    
    def update_conditions(self,conditions,numerics,unknowns):
        """ Segment.update_conditions(conditions,numerics,unknowns)
            if needed, updates the conditions given the current free unknowns and numerics
            called once per segment solver iteration
        """
        
        # unpack unknowns
        throttle = unknowns.controls.throttle
        theta    = unknowns.controls.theta
        psi      = unknowns.controls.psi 
        
        # apply unknowns
        conditions.propulsion.throttle[:,0]            = throttle[:,0]
        conditions.frames.body.inertial_rotations[:,1] = theta[:,0]
        #conditions.frames.body.inertial_rotations[:,2] = psi[:,0]
        
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
        a  = conditions.frames.inertial.acceleration_vector
        m  = conditions.weights.total_mass[:,0] 
        
        # process
        residuals.controls.Fx[:,0] = FT[:,0] - a[:,0]*m
        residuals.controls.Fy[:,0] = FT[:,1] - a[:,1]*m
        residuals.controls.Fz[:,0] = FT[:,2] - a[:,2]*m
        
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