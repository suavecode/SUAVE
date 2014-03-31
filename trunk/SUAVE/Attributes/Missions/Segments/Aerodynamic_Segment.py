# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure                    import Data, Data_Exception
from SUAVE.Structure                    import Container as ContainerBase
from SUAVE.Attributes.Planets           import Planet
from SUAVE.Attributes.Atmospheres       import Atmosphere
from SUAVE.Methods.Utilities.Chebyshev  import chebyshev_data
from SUAVE.Methods.Utilities            import atleast_2d_col
from SUAVE.Geometry.Three_Dimensional   import angles_to_dcms, orientation_product

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

class Aerodynamic_Segment(Data):
    
    # ------------------------------------------------------------------
    #   Methods For Initialization
    # ------------------------------------------------------------------    
    
    def __defaults__(self):
        self.tag = 'Aerodynamic Segment'
        
        # presumably we fly in an atmosphere
        self.planet     = Planet()
        self.atmosphere = Atmosphere()        


        # --- Conditions and Unknowns
        
        # user shouldn't change these in an input script
        # only used for processing / post processing
        # they will be shared with analysis modules and meaningful naming is important
        
        # base matricies
        # use a trivial operation to copy the array
        ones_1col = np.ones([1,1])
        ones_2col = np.ones([1,2])
        ones_3col = np.ones([1,3])        
        
        
        # --- Conditions        
        
        # setup conditions
        conditions = Data()
        conditions.frames       = Data()
        conditions.freestream   = Data()
        conditions.aerodynamics = Data()
        conditions.propulsion   = Data()
        conditions.weights      = Data()
        conditions.engergy      = Data()
        self.conditions = conditions
        
        # inertial frame conditions
        conditions.frames.inertial = Data()        
        conditions.frames.inertial.position_vector      = ones_3col * 0
        conditions.frames.inertial.velocity_vector      = ones_3col * 0
        conditions.frames.inertial.acceleration_vector  = ones_3col * 0
        conditions.frames.inertial.gravity_force_vector = ones_3col * 0
        conditions.frames.inertial.total_force_vector   = ones_3col * 0
        conditions.frames.inertial.time                 = ones_1col * 0

        # wind frame conditions
        conditions.frames.wind = Data()        
        conditions.frames.wind.body_rotations           = ones_3col * 0
        conditions.frames.wind.velocity_vector          = ones_3col * 0
        conditions.frames.wind.lift_force_vector        = ones_3col * 0
        conditions.frames.wind.drag_force_vector        = ones_3col * 0
        conditions.frames.wind.transform_to_inertial    = np.empty([0,0,0])
        
        # body frame conditions
        conditions.frames.body = Data()        
        conditions.frames.body.inertial_rotations       = ones_3col * 0
        conditions.frames.body.thrust_force_vector      = ones_3col * 0
        conditions.frames.body.transform_to_inertial    = np.empty([0,0,0])
        
        # freestream conditions
        conditions.freestream.velocity           = ones_1col * 0
        conditions.freestream.mach_number        = ones_1col * 0
        conditions.freestream.pressure           = ones_1col * 0
        conditions.freestream.temperature        = ones_1col * 0
        conditions.freestream.density            = ones_1col * 0
        conditions.freestream.speed_of_sound     = ones_1col * 0
        conditions.freestream.viscosity          = ones_1col * 0
        conditions.freestream.altitude           = ones_1col * 0
        conditions.freestream.gravity            = ones_1col * 0
        conditions.freestream.reynolds_number    = ones_1col * 0
        conditions.freestream.dynamic_pressure   = ones_1col * 0
        
        # aerodynamics conditions
        conditions.aerodynamics.angle_of_attack  = ones_1col * 0
        conditions.aerodynamics.side_slip_angle  = ones_1col * 0
        conditions.aerodynamics.roll_angle       = ones_1col * 0
        conditions.aerodynamics.lift_coefficient = ones_1col * 0
        conditions.aerodynamics.drag_coefficient = ones_1col * 0
        conditions.aerodynamics.lift_breakdown   = Data()
        conditions.aerodynamics.drag_breakdown   = Data()
        
        # propulsion conditions
        conditions.propulsion.throttle           = ones_1col * 0
        conditions.propulsion.fuel_mass_rate     = ones_1col * 0
        conditions.propulsion.thrust_breakdown   = Data()
        
        # weights conditions
        conditions.weights.total_mass            = ones_1col * 0
        conditions.weights.weight_breakdown      = Data()
        
        # energy conditions
        conditions.energies.total_energy         = ones_1col * 0
        conditions.energies.total_efficiency     = ones_1col * 0
        conditions.energies.gravity_energy       = ones_1col * 0
        conditions.energies.propulusion_power    = ones_1col * 0
        
        # --- Unknowns        
        
        # setup unknowns
        unknowns = Data()
        unknowns.states   = Data()
        unknowns.controls = Data()
        unknowns.finals   = Data()
        self.unknowns = unknowns
        
        # an example
        ## unknowns.states.gamma = ones_1col + 0
        
        
        # --- Numerics
        
        # numerical options
        self.options.tag = 'Solution Options'
        self.options.n_control_points              = 16
        self.options.jacobian                      = "complex"
        self.options.tolerance_solution            = 1e-8
        self.options.tolerance_boundary_conditions = 1e-8        
        
        # differentials
        self.differentials.method = chebyshev_data
        
        return
    
    
    def check(self):
        """ Segment.check():
            error checking of segment inputs
        """
        
        ## CODE
        
        return
    
    def initialize_conditions(self,conditions):
        """ Segment.initialize_conditions(conditions)
            update the segment conditions
            pin down as many condition variables as possible in this function
            
            Inputs:
                conditions - the conditions data dictionary, with initialized zero arrays, 
                             with number of rows = segment.conditions.n_control_points
                
            Outputs:
                conditions - the conditions data dictionary, updated with the 
                             values that can be precalculated
            
            Assumptions:
                --
                
            Usage Notes:
                sill need to inspect segment (self) for user inputs
                will be called before solving the segments free unknowns
                
        """
        
        # unpack inputs
        ## CODE
        
        # setup
        ## CODE
        
        # process
        ## CODE
        
        # pack outputs
        ## CODE
        
        return conditions
    
    
    # ------------------------------------------------------------------
    #   Methods For Solver Iterations
    # ------------------------------------------------------------------    
    
    def update_conditions(unknowns,conditions,differentials):
        """ Segment.update_conditions(unknowns, conditions, differentials)
            if needed, updates the conditions given the current free unknowns and differentials
            called once per segment solver iteration
            
            Inputs - 
                unknowns      - data dictionary of segment free unknowns with fields:
                    states, controls, finals
                    these are defined in segment.__defaults__
                conditions    - data dictionary of segment conditions
                    these are defined in segment.__defaults__
                differentials - data dictionary of differential operators for this iteration
                
            Outputs - 
                conditions - data dictionary of update conditions
                
            Assumptions - 
                preserves the shapes of arrays in conditions

        """
        
        # unpack inputs
        ## CODE
        
        # setup
        ## CODE
        
        # process
        ## CODE
        
        # pack outputs
        ## CODE
        
        return residuals
    
    def solve_residuals(unknowns,conditions,differentials):
        """ Segment.solve_residuals(unknowns, conditions, differentials)
            the hard work, solves the residuals for the free unknowns
            called once per segment solver iteration
            
            Inputs - 
                unknowns      - data dictionary of segment free unknowns with fields:
                    states, controls, finals
                    these are defined in segment.__defaults__
                conditions    - data dictionary of segment conditions
                    these are defined in segment.__defaults__
                differentials - data dictionary of differential operators for this iteration
                
            Outputs - 
                residuals - data dictionary of residuals, same dictionary structure as unknowns
            
            Usage Notes - 
                after this method, residuals composed into a final residual vector:
                    R = [ [ d(unknowns.states)/dt - residuals.states   ] ;
                          [                         residuals.controls ] ;
                          [                         residuals.finals   ] ] = [0] ,
                    where the segment solver will find a root of R = [0]

        """
        
        # unpack inputs
        ## CODE
        
        # setup
        ## CODE
        
        # process
        ## CODE
        
        # pack outputs
        ## CODE
        
        return residuals
    
    
    # ------------------------------------------------------------------
    #   Methods For Post-Solver
    # ------------------------------------------------------------------    
    
    def post_process(self,unknowns,conditions,differentials):
        """ Segment.post_process(unknowns, conditions, differentials)
            post processes the conditions after converging the segment solver
            
            Inputs - 
                unknowns - data dictionary of converged segment free unknowns with fields:
                    states, controls, finals
                    these are defined in segment.__defaults__
                conditions - data dictionary of segment conditions
                    these are defined in segment.__defaults__
                differentials - data dictionary of the converged differential operators
                
            Outputs - 
                conditions - data dictionary with additional post-processed data
            
            Usage Notes - 
                use this to store the unknowns and any other interesting in conditions
                    for later plotting
            
        """
        
        # unpack inputs
        ## CODE
        
        # setup
        ## CODE
        
        # process
        ## CODE
        
        # pack outputs
        ## CODE
        
        return

    
    # ----------------------------------------------------------------------
    #  Segment Helper Methods
    # ----------------------------------------------------------------------
    
    def compute_atmosphere(self,conditions,atmosphere):
    
        # unpack
        h = conditions.freestream.altitude
    
        # compute
        p, T, rho, a, mew = atmosphere.compute_values(h)
        
        # pack
        conditions.freestream.pressure       = p
        conditions.freestream.temperature    = T
        conditions.freestream.density        = rho
        conditions.freestream.speed_of_sound = a
        conditions.freestream.viscosity      = mew
    
        return conditions
    
    def compute_gravity(self,conditions,planet):
    
        # unpack
        g0 = planet.sea_level_gravity       # m/s^2
        
        # calculate
        g = g0        # m/s^2 (placeholder for better g models)
        
        # pack
        conditions.freestream.gravity = g
        
        return conditions
    
    def compute_freestream(self,conditions):
        """ compute_freestream(condition)
            computes freestream values
            
            Inputs:
            
            Outputs:
            dynamic pressure
            mach number
            reynolds number - DIMENSIONAL - PER UNIT LENGTH - MUST MULTIPLY BY REFERENCE LENGTH
            
        """
        
        # unpack
        Vvec = conditions.frames.inertial.velocity_vector
        rho  = conditions.freestream.density
        a    = conditions.freestream.speed_of_sound 
        mew  = conditions.freestream.viscosity 
        
        # velocity magnitude
        Vmag2 = np.sum( V**2, axis=1)[:,None] # keep 2d column vector
        Vmag  = np.sqrt(V2)
    
        # dynamic pressure
        q = 0.5 * rho * Vmag2 # Pa
        
        # Mach number
        M = Vmag / a
    
        # Reynolds number
        Re = rho * Vmag / mew  # per m
        
        # pack
        conditions.freestream.velocity         = Vmag
        conditions.freestream.mach_number      = M
        conditions.freestream.reynolds_number  = Re
        conditions.freestream.dynamic_pressure = q
        
        return conditions
    
    def compute_aerodynamics(self,aerodynamics_model,conditions):
        """ compute_aerodynamics()
            gets aerodynamics conditions
            
            Outputs - 
                lift, drag coefficient, lift drag force, stores to body axis data
            
            Assumptions -
                +X out nose
                +Y out starboard wing
                +Z down
            
        """
        
        # unpack
        q    = conditions.freestream.dynamic_pressure
        Sref = aerodynamics_model.reference_area # TODO - where???
        
        # call aerodynamics model
        results = aerodynamics_model(conditions)     # nondimensional
        
        # unpack results
        CL = atleast_2d_col( results.lift_coefficient )
        CD = atleast_2d_col( results.drag_coefficient )
        
        # compute forces
        f_aero = q * Sref
        FL = -CL * f_aero
        FD = -CD * f_aero
        
        # pack conditions
        conditions.aerodynamics.lift_coefficient[:,0] = CL[:,0]
        conditions.aerodynamics.drag_coefficient[:,0] = CD[:,0]
        conditions.frames.wind.lift_force_vector[:,3] = -FL[:,0] # z-axis
        conditions.frames.wind.drag_force_vector[:,0] = -FD[:,0] # x-axis
        
        return conditions
    
    def compute_propulsion(self,propulsion_model,conditions):
        
        # unpack
        throttle         = conditions.propulsion.throttle
        
        # call propulsion model
        results = propulsion_model(conditions)
        
        # unpack results
        F    = results.thrust_force
        mdot = atleast_2d_col( results.fuel_mass_rate )
        P    = atleast_2d_col( results.thurst_power   )
        
        # pack conditions
        conditions.frames.body.thrust_force_vector[:,:] = F[:,:]
        conditions.propulsion.fuel_mass_rate[:,0]       = -mdot[:,0]
        conditions.energies.propulusion_power[:,0]      = P[:,0]
    
        return conditions
    
    def compute_weights(self,conditions,differentials):
        
        # unpack
        m0        = conditions.weights.total_mass[0,0]
        mdot_fuel = conditions.propulsion.fuel_mass_rate
        I         = differentials.I
        g         = conditions.freestream.gravity
    
        # calculate
        m = m0 + np.dot(I,mdot_fuel)
        W = m*g
        
        # pack
        conditions.weights.total_mass[1:,0] = m[1:,0] # don't mess with m0
        conditions.frames.inertial.gravity_force_vector = W
    
        return conditions
    
        
    def compute_orientations(self,conditions):
        
        # unpack
        V_inertial = conditions.frames.inertial.velocity_vector
        
        body_inertial_rotations = conditions.frames.body.inertial_rotations
        psi   = body_inertial_rotations[:,0,None]
        theta = body_inertial_rotations[:,1,None]
        phi   = body_inertial_rotations[:,2,None]
        
        # body frame tranformation matrices
        T_inertial2body = angles_to_dcms(body_inertial_rotations,'ZYX')
        T_body2inertial = np.einsum('aji',T_inertial2body)
        
        # transform V_I to body frame
        V_body = orientation_product(T_inertial2body,V_inertial)
        
        # project V_I into body x-z plane
        V_stability = V_body
        V_stability[:,1] = 0
        V_stability_magnitude = np.sqrt( np.sum(V_stability**2,axis=1) )
        V_stability_direction = V_stability / V_stability_magnitude
        
        # calculate angle of attack
        alpha = np.arctan2(V_stability[:,2],V_stability[:,0])[:,None]
        
        # calculate side slip
        beta = np.arctan2(V_stability[:,1],V_stability_direction)[:,None]
        
        # wind frame rotations
        wind_body_rotations = body_inertial_rotations * 0.
        wind_body_rotations[:,0] = 0 
        wind_body_rotations[:,1] = alpha[:,0]
        wind_body_rotations[:,2] = beta[:,0]
        
        # wind frame tranformation matricies
        T_wind2body = angles_to_dcms(wind_body_rotations,'ZYX')
        T_body2wind = np.einsum('aji',T_wind2body)
        T_wind2inertial = orientation_product(T_wind2body,T_body2inertial)
        
        # pack aero
        conditions.aerodynamics.angle_of_attack[:,0] = alpha[:,0]
        conditions.aerodynamics.side_slip_angle[:,0] = beta[:,0]
        conditions.aerodynamics.roll_angle[:,0]      = phi[:,0]
        
        # pack wind rotations
        conditions.frames.wind.body_rotations = wind_body_rotations
        
        # pack transformation matricies
        conditions.frames.wind.transform_to_inertial = T_wind2inertial
        conditions.frames.body.transform_to_inertial = T_body2inertial
        
        return conditions
        
    def compute_forces(self,conditions):
        
        # unpack forces
        wind_lift_force_vector        = conditions.frames.wind.lift_force_vector
        wind_drag_force_vector        = conditions.frames.wind.drag_force_vector
        body_thrust_force_vector      = conditions.frames.body.thrust_force_vector
        inertial_gravity_force_vector = conditions.frames.inertial.gravity_force_vector
        
        # unpack transformation matrices
        T_body2inertial = conditions.frames.body.transform_to_inertial
        T_wind2inertial = conditions.frames.wind.transform_to_inertial
        
        # to inertial frame
        L = orientation_product(T_wind2inertial,wind_lift_force_vector)
        D = orientation_product(T_wind2inertial,wind_drag_force_vector)
        T = orientation_product(T_body2inertial,body_thrust_force_vector)
        W = inertial_gravity_force_vector
        
        # sum of the forces
        F = L + D + T + W
        # like a boss
        
        # pack
        conditions.frames.inertial.total_force_vector[:,:] = F[:,:]
        
        return conditions
          

# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

class Container(ContainerBase):
    pass

Aerodynamic_Segment.Container = Container
