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
from SUAVE.Geometry.Three_Dimensional   import angles_to_dcms, orientation_product, orientation_transpose
from Base_Segment import Base_Segment

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

class Aerodynamic_Segment(Base_Segment):
    
    # ------------------------------------------------------------------
    #   Methods For Initialization
    # ------------------------------------------------------------------    
    
    def __defaults__(self):
        self.tag = 'Aerodynamic Segment'
        
        # atmosphere and planet
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
        conditions.energies     = Data()
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
        
        return
    
    
    
    # ------------------------------------------------------------------
    #   Methods For Solver
    # ------------------------------------------------------------------        

    # See:
    #   Base_Segment.check_inputs()
    #   Base_Segment.initialize_conditions()
    #   Base_Segment.update_conditions()
    #   Base_Segment.solve_residuals()
    #   Base_Segment.post_process()
    
    def update_conditions(self,conditions,numerics,unknowns):
        
        # unpack models
        aero_model = self.config.aerodynamics_model
        prop_model = self.config.propulsion_model        
        
        # angle of attacks
        conditions = self.compute_orientations(conditions)
        
        # aerodynamics
        conditions = self.compute_aerodynamics(aero_model,conditions)
        
        # propulsion
        conditions = self.compute_propulsion(prop_model,conditions,numerics)
        
        # weights
        conditions = self.compute_weights(conditions,numerics)
        
        # total forces
        conditions = self.compute_forces(conditions)
        
        return conditions
    
    
    # ----------------------------------------------------------------------
    #  Segment Helper Methods
    # ----------------------------------------------------------------------
    
    def compute_atmosphere(self,conditions,atmosphere):
        """ Aerodynamic_Segment.compute_atmosphere(conditions,atmosphere)
            computes conditions of the atmosphere at given altitudes
            
            Inputs:
                conditions - data dictionary with ...
                    freestream.altitude
                atmoshere - an atmospheric model
            Outputs:
                conditions - with...
                    freestream.pressure
                    freestream.temperature
                    freestream.density
                    freestream.speed_of_sound
                    freestream.viscosity
                    
        """
        
    
        # unpack
        h = conditions.freestream.altitude
    
        # compute
        p, T, rho, a, mew = atmosphere.compute_values(h)
        
        # pack
        conditions.freestream.pressure[:,0]       = p
        conditions.freestream.temperature[:,0]    = T
        conditions.freestream.density[:,0]        = rho
        conditions.freestream.speed_of_sound[:,0] = a
        conditions.freestream.viscosity[:,0]      = mew
    
        return conditions
    
    
    def compute_gravity(self,conditions,planet):
    
        # unpack
        g0 = planet.sea_level_gravity       # m/s^2
        
        # calculate
        g = g0        # m/s^2 (placeholder for better g models)
        
        # pack
        conditions.freestream.gravity[:,0] = g
        
        return conditions
    
    
    def compute_freestream(self,conditions):
        """ compute_freestream(condition)
            computes freestream values
            
            Inputs:
                conditions - data dictionary with fields...
                    frames.inertial.velocity_vector
                    freestream.density
                    freestream.speed_of_sound
                    freestream.viscosity
                    
            Outputs:
                conditions with fields:
                    freestream.dynamic pressure
                    freestream.mach number
                    freestream.reynolds number - DIMENSIONAL - PER UNIT LENGTH - MUST MULTIPLY BY REFERENCE LENGTH
            
        """
        
        # unpack
        Vvec = conditions.frames.inertial.velocity_vector
        rho  = conditions.freestream.density
        a    = conditions.freestream.speed_of_sound 
        mew  = conditions.freestream.viscosity 
        
        # velocity magnitude
        Vmag2 = np.sum( Vvec**2, axis=1)[:,None] # keep 2d column vector
        Vmag  = np.sqrt(Vmag2)
    
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
            
            Inputs - 
                aerodynamics_model - a callable that will recieve ...
                conditions         - passed directly to the aerodynamics model
            
            Outputs - 
                lift, drag coefficient, lift drag force, stores to body axis data
            
            Assumptions -
                +X out nose
                +Y out starboard wing
                +Z down
            
        """
        
        # call aerodynamics model
        results = aerodynamics_model( conditions )     # nondimensional
        
        # unpack results
        L = results.lift_force_vector
        D = results.drag_force_vector
                
        # pack conditions
        conditions.frames.wind.lift_force_vector[:,:] = L[:,:] # z-axis
        conditions.frames.wind.drag_force_vector[:,:] = D[:,:] # x-axis
        
        return conditions
    
    
    def compute_propulsion(self,propulsion_model,conditions,numerics):
        """ compute_propulsion()
            gets propulsion conditions
            
            Inputs - 
                propulsion_model - a callable that will recieve ...
                conditions         - passed directly to the propulsion model
            
            Outputs - 
                results - a data dictionary with ...
                    thrust_force   - a 3-column array with rows of total thrust force vectors
                        for each control point, in the body frame
                    fuel_mass_rate - the total fuel mass flow rate for each control point
                    thrust_power   - the total propulsion power for each control point
            
            Assumptions -
                +X out nose
                +Y out starboard wing
                +Z down
            
        """
        
        # for current propulsion models
        ## TODO: update propulsion modules
        
        N = self.numerics.n_control_points
        
        eta = conditions.propulsion.throttle[:,0]
        
        #state = Data()
        #state.q  = conditions.freestream.dynamic_pressure[:,0]
        #state.g0 = conditions.freestream.gravity[:,0]
        #state.V  = conditions.freestream.velocity[:,0]
        #state.M  = conditions.freestream.mach_number[:,0]
        #state.T  = conditions.freestream.temperature[:,0]
        #state.p  = conditions.freestream.pressure[:,0]
        
        F, mdot, P = propulsion_model(eta, conditions)
        
        F_vec = np.zeros([N,3])
        F_vec[:,0] = F[:]
        mdot = atleast_2d_col( mdot )
        P    = atleast_2d_col( P    )
        
        ## TODO ---
        ## call propulsion model
        #results = propulsion_model( conditions )
        
        ## unpack results
        #F    = results.thrust_force
        #mdot = atleast_2d_col( results.fuel_mass_rate )
        #P    = atleast_2d_col( results.thurst_power   )
        
        # pack conditions
        conditions.frames.body.thrust_force_vector[:,:] = F_vec[:,:]
        conditions.propulsion.fuel_mass_rate[:,0]       = mdot[:,0]
        conditions.energies.propulusion_power[:,0]      = P[:,0]
        
        return conditions
    
    
    def compute_weights(self,conditions,numerics):
        
        # unpack
        m0        = conditions.weights.total_mass[0,0]
        m_empty   = self.config.Mass_Props.m_empty
        mdot_fuel = conditions.propulsion.fuel_mass_rate
        I         = numerics.integrate_time
        g         = conditions.freestream.gravity
        
        # calculate
        m = m0 + np.dot(I, -mdot_fuel )
        
        # feasibility constraint
        m[ m < m_empty ] = m_empty
        
        # weight
        W = m*g
        
        # pack
        conditions.weights.total_mass[1:,0] = m[1:,0] # don't mess with m0
        conditions.frames.inertial.gravity_force_vector[:,2] = W[:,0]
        
        return conditions
    
        
    def compute_orientations(self,conditions):
        
        # unpack
        V_inertial = conditions.frames.inertial.velocity_vector
        body_inertial_rotations = conditions.frames.body.inertial_rotations
        
        # --- Body Frame
        
        # body frame rotations
        psi   = body_inertial_rotations[:,0,None]
        theta = body_inertial_rotations[:,1,None]
        phi   = body_inertial_rotations[:,2,None]
        
        # body frame tranformation matrices
        T_inertial2body = angles_to_dcms(body_inertial_rotations,'ZYX')
        T_body2inertial = orientation_transpose(T_inertial2body)
        
        # transform inertial velocity to body frame
        V_body = orientation_product(T_inertial2body,V_inertial)
        
        # project inertial velocity into body x-z plane
        V_stability = V_body
        V_stability[:,1] = 0
        V_stability_magnitude = np.sqrt( np.sum(V_stability**2,axis=1) )[:,None]
        #V_stability_direction = V_stability / V_stability_magnitude
        
        # calculate angle of attack
        alpha = np.arctan2(V_stability[:,2],V_stability[:,0])[:,None]
        
        # calculate side slip
        beta = np.arctan2(V_body[:,1],V_stability_magnitude[:,0])[:,None]
        
        # pack aerodynamics angles
        conditions.aerodynamics.angle_of_attack[:,0] = alpha[:,0]
        conditions.aerodynamics.side_slip_angle[:,0] = beta[:,0]
        conditions.aerodynamics.roll_angle[:,0]      = phi[:,0]        
        
        # pack transformation tensor
        conditions.frames.body.transform_to_inertial = T_body2inertial
        
        
        # --- Wind Frame
        
        # back calculate wind frame rotations
        wind_body_rotations = body_inertial_rotations * 0.
        wind_body_rotations[:,0] = 0 
        wind_body_rotations[:,1] = alpha[:,0]
        wind_body_rotations[:,2] = beta[:,0]
        
        # wind frame tranformation matricies
        T_wind2body = angles_to_dcms(wind_body_rotations,'ZYX')
        T_body2wind = orientation_transpose(T_wind2body)
        T_wind2inertial = orientation_product(T_wind2body,T_body2inertial)
        
        # pack wind rotations
        conditions.frames.wind.body_rotations = wind_body_rotations
        
        # pack transformation tensor
        conditions.frames.wind.transform_to_inertial = T_wind2inertial
        
        
        # done!
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
