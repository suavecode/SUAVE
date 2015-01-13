# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from SUAVE.Attributes.Missions.Segments import Aerodynamic_Segment, Base_Segment
from SUAVE.Core import Data
from SUAVE.Geometry.Three_Dimensional   import orientation_product
from SUAVE.Attributes import Units

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Ground_Segment(Aerodynamic_Segment):
    """
        Base segment for takeoff and landing segments. Integrates equations of motion
        including rolling friction.

        Notes Regarding Friction Coefficients
        Dry asphalt or concrete: .04 brakes off, .4 brakes on
        Wet asphalt or concrete: .05 brakes off, .225 brakes on
        Icy asphalt or concrete: .02 brakes off, .08 brakes on
        Hard turf:               .05 brakes off, .4 brakes on
        Firm dirt:               .04 brakes off, .3 brakes on
        Soft turf:               .07 brakes off, .2 brakes on
        Wet grass:               .08 brakes off, .2 brakes on
        FROM: General Aviation Aircraft Design: Applied Methods and Procedures,
        by Snorri Gudmundsson, copyright 2014, published by Elsevier, Waltham,
        MA, USA [p.938]
    """

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  

    def __defaults__(self):

        self.tag = 'Ground Segment'

        self.ground_incline       = 0.0
        self.friction_coefficient = 0.04
        self.throttle             = None
        self.velocity_start       = 0.0
        self.velocity_end         = 0.0
        self.battery_energy = 0.0
        self.latitude       = 0.0
        self.longitude      = 0.0        

        #Initialize data structure
        # base matricies
        # use a trivial operation to copy the array
        ones_1col = np.ones([1,1])
        ones_2col = np.ones([1,2])
        ones_3col = np.ones([1,3]) 

        conditions = self.conditions

        # --- Properties specific to ground segments
        conditions.ground = Data()
        conditions.ground.incline                      = ones_1col * 0.0
        conditions.ground.friction_coefficient         = ones_1col * 0.0
        conditions.frames.inertial.ground_force_vector = ones_3col * 0.0

        # --- Unknowns
        unknowns = self.unknowns
        unknowns.controls.velocity_x = ones_1col
        unknowns.finals.time         = ones_1col

        # --- Residuals
        residuals = self.residuals
        residuals.controls.acceleration_x       = ones_1col
        residuals.finals.final_velocity_error   = ones_1col

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

    def initialize_arrays(self,conditions,numerics,unknowns,residuals):
        """ Modified initialize_arrays method to remove the initial condition from
            unknowns.controls.velocity_x (and the corresponding acceleration residual)
        """

        # run the superclass array initialization
        unknowns, conditions, residuals = \
            Aerodynamic_Segment.initialize_arrays(self, conditions, numerics, unknowns, residuals)

        # modify unknowns and residuals to prevent solver from changing v0
        unknowns.controls.velocity_x      = unknowns.controls.velocity_x[1:,0]
        residuals.controls.acceleration_x = residuals.controls.acceleration_x[1:,0]

        return unknowns, conditions, residuals

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

        conditions = Aerodynamic_Segment.initialize_conditions(self,conditions,numerics,initials)

        # unpack inputs
        v0       = self.velocity_start
        vf       = self.velocity_end
        alt      = self.altitude
        atmo     = self.atmosphere
        planet   = self.planet
        conditions.ground.incline[:,0]              = self.ground_incline
        conditions.ground.friction_coefficient[:,0] = self.friction_coefficient
        N        = len(conditions.frames.inertial.velocity_vector[:,0])

        # avoid having zero velocity since aero and propulsion models need non-zero Reynolds number
        if v0 == 0.0: v0 = 0.01
        if vf == 0.0: vf = 0.01

        # repack
        self.velocity_start = v0
        self.velocity_end   = vf

        # pack conditions
        conditions.frames.inertial.velocity_vector[:,0] = np.linspace(v0,vf,N)
        conditions.freestream.altitude[:,0]             = alt

        # freestream atmosphereric conditions
        conditions = self.compute_atmosphere(conditions,atmo)
        conditions = self.compute_gravity(conditions,planet)

        # done
        return conditions


    # ------------------------------------------------------------------
    #   Methods For Iterations
    # ------------------------------------------------------------------ 

    def update_conditions(self,conditions,numerics,unknowns):
        """ Compute orientation, and force conditions. Modified from
            originial Aerodynamic_Segment to include rolling friction
        """
        # unpack unknowns
        velocity_x   = unknowns.controls.velocity_x
        final_time   = unknowns.finals.time

        #apply unknowns
        conditions.frames.inertial.velocity_vector[1:,0] = velocity_x

        # unpack models
        aero_model = self.config.aerodynamics_model
        prop_model = self.config.propulsion_model

        # freestream conditions
        conditions = self.compute_freestream(conditions)

        # call Aerodynamic_Segment's update_conditions
        conditions = Aerodynamic_Segment.update_conditions(self,conditions,numerics,unknowns)

        # rolling friction
        conditions = self.compute_ground_forces(conditions)

        # total forces
        conditions = self.compute_forces(conditions)	

        return conditions



    def compute_ground_forces(self,conditions):
        """ Compute the rolling friction on the aircraft """

        # unpack
        W                      = conditions.frames.inertial.gravity_force_vector[:,2,None]
        friction_coeff         = conditions.ground.friction_coefficient
        wind_lift_force_vector = conditions.frames.wind.lift_force_vector

        #transformation matrix to get lift in inertial frame
        T_wind2inertial = conditions.frames.wind.transform_to_inertial

        # to inertial frame
        L = orientation_product(T_wind2inertial,wind_lift_force_vector)[:,2,None]

        #compute friction force
        N  = -(W + L)
        Ff = N * friction_coeff

        #pack results. Friction acts along x-direction
        conditions.frames.inertial.ground_force_vector[:,2] = N[:,0]
        conditions.frames.inertial.ground_force_vector[:,0] = Ff[:,0]

        return conditions


    def compute_forces(self,conditions):

        conditions = Aerodynamic_Segment.compute_forces(self,conditions)

        # unpack forces
        total_aero_forces = conditions.frames.inertial.total_force_vector
        inertial_ground_force_vector = conditions.frames.inertial.ground_force_vector

        # sum of the forces, including friction force
        F = total_aero_forces + inertial_ground_force_vector
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
        D  = numerics.differentiate_time
        m  = conditions.weights.total_mass

        # process and pack
        acceleration = np.dot(D , v)
        conditions.frames.inertial.acceleration_vector = acceleration

        residuals.finals.final_velocity_error = (v[-1,0] - vf)
        residuals.controls.acceleration_x[:] = (FT / m - acceleration)[1:,0]

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