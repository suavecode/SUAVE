## @ingroup Analyses-Mission-Segments-Climb
# Optimized.py
#
# Created:  Mar 2016, E. Botero 
#           Apr 2020, M. Clarke
#           Aug 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses.Mission.Segments import Aerodynamic
from SUAVE.Analyses.Mission.Segments import Conditions

from SUAVE.Methods.Missions import Segments as Methods
from SUAVE.Methods.skip import skip

from SUAVE.Analyses import Process

# Units
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Climb
class Optimized(Aerodynamic):
    """ Optimize your climb segment. This is useful if you're not sure how your vehicle should climb.
        You can set any conditions parameter as the objective, for example setting time to climb or vehicle mass:
        segment.objective       = 'conditions.weights.total_mass[-1,0]'
        
        The ending airspeed is an optional parameter.
        
        This segment takes far longer to run than a normal segment. Wrapping this into a vehicle optimization
        has not yet been tested for robustness.
        
    
        Assumptions:
        Can use SNOPT if you have it installed through PyOpt. But defaults to SLSQP through 
        Runs a linear true airspeed mission first to initialize conditions.
        
        Source:
        None
    """          
    
    def __defaults__(self):
        """ This sets the default solver flow. Anything in here can be modified after initializing a segment.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """          
        
        # --------------------------------------------------------------
        #   User inputs
        # --------------------------------------------------------------
        self.altitude_start         = None
        self.altitude_end           = None
        self.air_speed_start        = None
        self.air_speed_end          = None
        self.objective              = None # This will be a key
        self.minimize               = True
        self.lift_coefficient_limit = 1.e20 
        self.seed_climb_rate        = 100. * Units['feet/min']
        self.algorithm              = 'SLSQP'
        
        
        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
        
        # conditions
        self.state.conditions.update( Conditions.Aerodynamics() )
        # initials and unknowns
        ones_row    = self.state.ones_row
        self.state.unknowns.throttle          = ones_row(1) * 0.8
        self.state.unknowns.body_angle        = ones_row(1) * 5.0 * Units.degrees
        self.state.unknowns.flight_path_angle = ones_row(1) * 3.0 * Units.degrees
        self.state.unknowns.velocity          = ones_row(1) * 1.0
        self.state.residuals.forces           = ones_row(2) * 0.0
        self.state.inputs_last                = None
        self.state.objective_value            = 0.0
        self.state.constraint_values          = 0.0
         
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize - before iteration
        # --------------------------------------------------------------
        initialize = self.process.initialize
        initialize.expand_state            = Methods.expand_state
        initialize.solved_mission          = Methods.Climb.Optimized.solve_linear_speed_constant_rate
        initialize.differentials           = Methods.Common.Numerics.initialize_differentials_dimensionless
        initialize.conditions              = skip

        # --------------------------------------------------------------
        #   Converge - starts iteration
        # --------------------------------------------------------------
        converge = self.process.converge
        
        converge.converge_root             = Methods.converge_opt    

        # --------------------------------------------------------------
        #   Iterate - this is iterated
        # --------------------------------------------------------------
        iterate = self.process.iterate
                
        # Update Initials
        iterate.initials = Process()
        iterate.initials.time              = Methods.Common.Frames.initialize_time
        iterate.initials.weights           = Methods.Common.Weights.initialize_weights
        iterate.initials.inertial_position = Methods.Common.Frames.initialize_inertial_position
        iterate.initials.planet_position   = Methods.Common.Frames.initialize_planet_position
        
        # Unpack Unknowns
        iterate.unknowns = Process()
        iterate.unknowns.mission           = Methods.Climb.Optimized.unpack_unknowns
        
        # Update Conditions
        iterate.conditions = Process()
        iterate.conditions.differentials   = Methods.Climb.Optimized.update_differentials
        iterate.conditions.acceleration    = Methods.Common.Frames.update_acceleration
        iterate.conditions.altitude        = Methods.Common.Aerodynamics.update_altitude
        iterate.conditions.atmosphere      = Methods.Common.Aerodynamics.update_atmosphere
        iterate.conditions.gravity         = Methods.Common.Weights.update_gravity
        iterate.conditions.freestream      = Methods.Common.Aerodynamics.update_freestream
        iterate.conditions.orientations    = Methods.Common.Frames.update_orientations 
        iterate.conditions.propulsion      = Methods.Common.Energy.update_thrust        
        iterate.conditions.aerodynamics    = Methods.Common.Aerodynamics.update_aerodynamics
        iterate.conditions.stability       = Methods.Common.Aerodynamics.update_stability
        iterate.conditions.weights         = Methods.Common.Weights.update_weights
        iterate.conditions.forces          = Methods.Common.Frames.update_forces
        iterate.conditions.planet_position = Methods.Common.Frames.update_planet_position

        # Solve Residuals
        iterate.residuals = Process()     
        iterate.residuals.total_forces     = Methods.Climb.Common.residual_total_forces
        
        # Set outputs
        iterate.outputs = Process()   
        iterate.outputs.objective          = Methods.Climb.Optimized.objective
        iterate.outputs.constraints        = Methods.Climb.Optimized.constraints
        iterate.outputs.cache_inputs       = Methods.Climb.Optimized.cache_inputs
        
        # --------------------------------------------------------------
        #   Finalize - after iteration
        # --------------------------------------------------------------
        finalize = self.process.finalize
        
        # Post Processing
        finalize.post_process = Process()        
        finalize.post_process.inertial_position = Methods.Common.Frames.integrate_inertial_horizontal_position
        finalize.post_process.stability         = Methods.Common.Aerodynamics.update_stability
        finalize.post_process.aero_derivatives  = skip
        finalize.post_process.noise             = Methods.Common.Noise.compute_noise
        
        return

