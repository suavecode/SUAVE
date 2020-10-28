## @ingroup Analyses-Mission-Segments-Transition
# Lift_Cruise_Optimized.py
#
# Created:  Apr 2018, M. Clarke
# Modified:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses.Mission.Segments import Aerodynamic
from SUAVE.Analyses.Mission.Segments import Conditions

from SUAVE.Methods.Missions import Segments as Methods

from SUAVE.Analyses import Process

# Units
from SUAVE.Core import Units
import SUAVE

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Transition
class Lift_Cruise_Optimized(Aerodynamic):
    """ Optimize your transition segment. This is useful if you're not sure how your vehicle should transition.
        You can set any conditions parameter as the objective, for example setting time to transition or vehicle mass
        
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
        self.altitude                = None
        self.air_speed_start         = None
        self.air_speed_end           = None
        self.acceleration            = None
        self.pitch_initial           = None
        self.pitch_final             = None     
        self.objective               = None # This will be a key
        self.minimize                = True
        self.lift_coefficient_limit  =  1.e20  
        self.algorithm               = 'SLSQP'
        
        
        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
        
        # conditions
        self.state.conditions.update( Conditions.Aerodynamics() )
        
        # initials and unknowns
        ones_row    = self.state.ones_row
        self.state.unknowns.rotor_power_coefficient = 0.05 * ones_row(1)
        self.state.unknowns.throttle_lift                    = 1.25 * ones_row(1)
        self.state.unknowns.propeller_power_coefficient      = 0.02 * ones_row(1)
        self.state.unknowns.throttle                         = .50 * ones_row(1)   
        self.state.residuals.network                         = 0.0 * ones_row(3)    
        self.state.residuals.forces                          = 0.0 * ones_row(2) 
        self.state.inputs_last                               = None
        self.state.objective_value                           = 0.0
        self.state.constraint_values                         = 0.0
         
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize - before iteration
        # --------------------------------------------------------------
        initialize = self.process.initialize
        initialize.expand_state            = Methods.expand_state
        initialize.solved_mission          = Methods.Transition.Lift_Cruise_Optimized.solve_constant_speed_constant_altitude_loiter
        initialize.differentials           = Methods.Common.Numerics.initialize_differentials_dimensionless
        initialize.conditions              = Methods.Transition.Constant_Acceleration_Constant_Pitchrate_Constant_Altitude.initialize_conditions

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
        iterate.unknowns.mission           = Methods.Transition.Lift_Cruise_Optimized.unpack_unknowns
        
        # Update Conditions
        iterate.conditions = Process()
        iterate.conditions.differentials   = Methods.Transition.Lift_Cruise_Optimized.update_differentials
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
        iterate.residuals.total_forces     = Methods.Transition.Constant_Acceleration_Constant_Pitchrate_Constant_Altitude.residual_total_forces

        
        # Set outputs
        iterate.outputs = Process()   
        iterate.outputs.objective          = Methods.Transition.Lift_Cruise_Optimized.objective
        iterate.outputs.constraints        = Methods.Transition.Lift_Cruise_Optimized.constraints
        iterate.outputs.cache_inputs       = Methods.Transition.Lift_Cruise_Optimized.cache_inputs
        
        # --------------------------------------------------------------
        #   Finalize - after iteration
        # --------------------------------------------------------------
        finalize = self.process.finalize
        
        # Post Processing
        finalize.post_process = Process()        
        finalize.post_process.inertial_position = Methods.Common.Frames.integrate_inertial_horizontal_position
        finalize.post_process.stability         = Methods.Common.Aerodynamics.update_stability
        
        return