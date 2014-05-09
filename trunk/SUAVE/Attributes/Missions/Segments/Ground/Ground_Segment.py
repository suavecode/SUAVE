# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from SUAVE.Attributes.Missions.Segments import Aerodynamic_Segment, Base_Segment
from SUAVE.Structure import Data
from SUAVE.Geometry.Three_Dimensional   import orientation_product

# import units
from SUAVE.Attributes import Units
km = Units.km
hr = Units.hr

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Ground_Segment(Aerodynamic_Segment):

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  
    
    def __defaults__(self):

        self.tag = 'Ground Segment'
       
        #Notes Regarding Friction Coefficients
		#  Dry asphalt or concrete: .04 brakes off, .4 brakes on
		#  Wet asphalt or concrete: .05 brakes off, .225 brakes on
		#  Icy asphalt or concrete: .02 brakes off, .08 brakes on
		#  Hard turf:               .05 brakes off, .4 brakes on
		#  Firm dirt:               .04 brakes off, .3 brakes on
		#  Soft turf:               .07 brakes off, .2 brakes on
		#  Wet grass:               .08 brakes off, .2 brakes on
	#FROM: General Aviation Aircraft Design: Applied Methods and Procedures,
        # by Snorri Gudmundsson, copyright 2014, published by Elsevier, Waltham,
        # MA, USA [p.938]
        self.ground_incline       = 0.0
        self.friction_coefficient = 0.04

        #Initialize data structure
        # base matricies
        # use a trivial operation to copy the array
        ones_1col = np.ones([1,1])
        ones_2col = np.ones([1,2])
        ones_3col = np.ones([1,3]) 

        conditions = self.conditions

        # --- Properties specific to ground segments
        conditions.ground                              = Data()
        conditions.ground.incline                      = 0.0 * Units.deg
        conditions.frames.inertial.ground_force_vector = ones_3col * 0

        # --- Unknowns
        unknowns = self.unknowns
        unknowns.states.velocity = np.ones([1,1])
        unknowns.finals.time     = np.ones([1,1])
        
        # --- Residuals
        residuals = self.residuals
        residuals.states.total_force_x  = np.ones([1,1])
        residuals.finals.velocity_error = np.ones([1,1])

	# Pack up outputs
        self.conditions = conditions
        self.unknowns   = unknowns
        self.residuals  = residuals

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
        """ Segment.initialize_conditions(conditions,numerics,initials=None)
            update the segment conditions
            pin down as many condition variables as possible in this function
            
            Inputs:
                conditions - the conditions data dictionary, with initialized
                zero arrays, with number of rows = 
                segment.conditions.n_control_points
                initials - a data dictionary with 1-row column arrays pulled from
                the last row of the previous segment's conditions data, or none
                if no previous segment
                
            Outputs:
                conditions - the conditions data dictionary, updated with the 
                             values that can be precalculated
            
            Assumptions:
                --
                
            Usage Notes:
                may need to inspect segment (self) for user inputs
                will be called before solving the segments free unknowns
                
        """

        # unpack inputs
        v0       = self.velocity_start + .001	#Non-zero velocity for aero and propulsion
        vf       = self.velocity_end
        alt      = self.altitude
        atmo     = self.atmosphere
        planet   = self.planet
        conditions.ground.incline = self.ground_incline * np.ones([1,1])
        conditions.ground.friction_coefficient = self.friction_coefficient * np.ones([1,1])
        
        # pack conditions
        conditions.frames.inertial.velocity_vector[:,0] = v0
        conditions.freestream.altitude[:,0]             = alt
        
        # freestream atmosphereric conditions
        conditions = self.compute_atmosphere(conditions,atmo)
        conditions = self.compute_gravity(conditions,planet)
        conditions = self.compute_freestream(conditions)        

        # done
        return conditions
    

    # ------------------------------------------------------------------
    #   Methods For Iterations
    # ------------------------------------------------------------------ 

    def update_velocity_vector(self,numerics,unknowns,conditions):
        """ helper function to set velocity_vector.
            called from update_differentials()
        """
        
        # unpack
        a_x   = conditions.frames.inertial.acceleration_vector[:,0][:,None]
        v_x_0 = conditions.frames.inertial.velocity_vector[0,0]
        I     = numerics.integrate_time

        # process
        v_x   = v_x_0 + np.dot(I, a_x )

        # pack
        conditions.frames.inertial.velocity_vector[:,0] = v_x[:,0]
        conditions.frames.wind.velocity_vector[:,0] = v_x[:,0]
        
        return conditions


    def update_conditions(self,conditions,numerics,unknowns):
        """ Compute orientation, and force conditions. Modified from
            originial Aerodynamic_Segment to include rolling friction
        """

        # unpack models
        aero_model = self.config.aerodynamics_model
        prop_model = self.config.propulsion_model        
        
        # angle of attacks
        conditions = self.compute_orientations(conditions)

        # freestream conditions
        conditions = self.compute_freestream(conditions)
        
        # aerodynamics
        conditions = self.compute_aerodynamics(aero_model,conditions)
        
        # propulsion
        conditions = self.compute_propulsion(prop_model,conditions)
        
        # weights
        conditions = self.compute_weights(conditions,numerics)

        # rolling friction
        conditions = self.compute_rolling_friction(conditions)

        # total forces
        conditions = self.compute_forces(conditions)
        
        # update and apply acceleration
        Force_x    = conditions.frames.inertial.total_force_vector
        mass       = conditions.weights.total_mass
        a_x        = Force_x / mass
        # pack acceleration and update velocity
        conditions.frames.inertial.acceleration_vector[:,0] = a_x[:,0]
        conditions = self.update_velocity_vector(numerics,unknowns,conditions)

        return conditions



    def compute_rolling_friction(self,conditions):
        """ Compute the rolling friction on the aircraft """

        # unpack
        W              = conditions.frames.inertial.gravity_force_vector[:,2]
        friction_coeff = conditions.ground.friction_coefficient
        wind_lift_force_vector        = conditions.frames.wind.lift_force_vector

        #transformation matrix to get lift in inertial frame
        T_wind2inertial = conditions.frames.wind.transform_to_inertial
        
        # to inertial frame
        L = orientation_product(T_wind2inertial,wind_lift_force_vector)[:,2]

        #compute friction force
        N  = -(W + L)
        Ff = N * friction_coeff

        #pack results. Friction acts along x-direction
        conditions.frames.inertial.ground_force_vector[:,2] = N
        conditions.frames.inertial.ground_force_vector[:,0] = Ff

        return conditions


    def compute_forces(self,conditions):

        # unpack forces
        wind_lift_force_vector        = conditions.frames.wind.lift_force_vector
        wind_drag_force_vector        = conditions.frames.wind.drag_force_vector
        body_thrust_force_vector      = conditions.frames.body.thrust_force_vector
        inertial_gravity_force_vector = conditions.frames.inertial.gravity_force_vector
        inertial_ground_force_vector  = conditions.frames.inertial.ground_force_vector  #Includes friction and normal force. Normal force will cancel out lift and weight
        
        # unpack transformation matrices
        T_body2inertial = conditions.frames.body.transform_to_inertial
        T_wind2inertial = conditions.frames.wind.transform_to_inertial
        
        # to inertial frame
        L = orientation_product(T_wind2inertial,wind_lift_force_vector)
        D = orientation_product(T_wind2inertial,wind_drag_force_vector)
        T = orientation_product(T_body2inertial,body_thrust_force_vector)
        W = inertial_gravity_force_vector
        R = inertial_ground_force_vector
        
        # sum of the forces, including friction force
        F = L + D + T + W + R
        # like a boss
        
        # pack
        conditions.frames.inertial.total_force_vector[:,:] = F[:,:]
        
        return conditions


    def solve_residuals(self,conditions,numerics,unknowns,residuals):
        """ Segment.solve_residuals(conditions,numerics,unknowns,residuals)
            the hard work, solves the residuals for the free unknowns
            called once per segment solver iteration
        """

        # unpack inputs
        FT = conditions.frames.inertial.total_force_vector
        vf = self.velocity_end
        v  = conditions.frames.inertial.velocity_vector
        m  = conditions.weights.total_mass
        a  = conditions.frames.inertial.acceleration_vector
        #print v[:,0]
        
        # process and pack
        residuals.states.total_force_x  = FT[:,0] - (np.transpose(m) * a[:,0])[0]
        residuals.finals.velocity_error = v[-1,0] - vf
        
        return residuals


    # ------------------------------------------------------------------
    #   Methods For Post-Solver
    # ------------------------------------------------------------------    
    
    def post_process(self,conditions,numerics,unknowns):
        """ Segment.post_process(conditions,numerics,unknowns)
            post processes the conditions after converging the segment solver.
	    Packs up the final position vector to allow estimation of the ground
	    roll distance (e.g., distance from brake release to rotation speed in
	    takeoff, or distance from touchdown to full stop on landing).
            
            Inputs - 
                unknowns - data dictionary of converged segment free unknowns with
	        fields:
                    states, controls, finals
                    these are defined in segment.__defaults__
                conditions - data dictionary of segment conditions
                    these are defined in segment.__defaults__
                numerics - data dictionary of the converged differential operators
                
            Outputs - 
                conditions - data dictionary with remaining fields filled with post-
	        processed conditions. Updated fields are:
		    conditions.frames.inertial.position_vector  (x-position update)
            
            Usage Notes - 
	        Use this to store the unknowns and any other interesting in 
		conditions for later plotting. For clarity and style, be sure to 
		define any new fields in segment.__defaults__
            
        """
        
        # unpack inputs
        ground_velocity  = conditions.frames.inertial.velocity_vector
        I                = numerics.integrate_time
        initial_position = conditions.frames.inertial.position_vector[0,:]
        
        # process
        position_vector = initial_position + np.dot( I , ground_velocity)
        
        # pack outputs
        conditions.frames.inertial.position_vector = position_vector
        
        
        return conditions