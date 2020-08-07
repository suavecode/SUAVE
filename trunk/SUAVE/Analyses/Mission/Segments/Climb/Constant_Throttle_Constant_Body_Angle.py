## @ingroup Analyses-Mission-Segments-Climb
# Constant_Throttle_Constant_Speed.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

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


# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Climb
class Constant_Throttle_Constant_Body_Angle(Aerodynamic):
    """ Climb at a constant throttle setting and true airspeed.
        This segment may not always converge as the vehicle could be deficient in thrust.
        Useful as a check to see the climb rate at the top of climb.
    
        Assumptions:
        You set a reasonable throttle setting that can provide enough thrust.
        
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
        self.altitude_start = None # Optional
        self.altitude_end   = 10. * Units.km
        self.velocity_x_start = None
        self.velocity_z_start = None
        self.throttle       = 0.5
        self.body_angle     = 0.
        
        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
        
        # conditions
        self.state.conditions.update( Conditions.Aerodynamics() )
        
        # initials and unknowns
        ones_row = self.state.ones_row
        ones_row_m1 = self.state.ones_row_m1
        self.state.unknowns.velocity_x  = ones_row_m1(1) * 100.
        self.state.unknowns.velocity_z  = ones_row_m1(1) * 100.
        #self.state.unknowns.wind_angle = ones_row(1) * 0.0 * Units.deg
        self.state.unknowns.time       = 5.
        self.state.residuals.force_x   = ones_row_m1(1) * 0.0     
        self.state.residuals.force_y   = ones_row_m1(1) * 0.0
        self.state.residuals.final_altitude = 0.0
        
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize - before iteration
        # --------------------------------------------------------------
        initialize = self.process.initialize
        
        initialize.expand_state            = Methods.expand_state
        initialize.differentials           = Methods.Common.Numerics.initialize_differentials_dimensionless
        initialize.conditions              = Methods.Climb.Constant_Throttle_Constant_Body_Angle.initialize_conditions
        initialize.velocities              = Methods.Climb.Constant_Throttle_Constant_Body_Angle.update_velocity_vector_from_wind_angle
        initialize.differentials_altitude  = Methods.Climb.Constant_Throttle_Constant_Body_Angle.update_differentials_altitude      
        
        # --------------------------------------------------------------
        #   Converge - starts iteration
        # --------------------------------------------------------------
        converge = self.process.converge
        
        converge.converge_root             = Methods.converge_root        
        
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
        iterate.unknowns.mission           = Methods.Climb.Constant_Throttle_Constant_Body_Angle.unpack_unknowns 
        
        # Update Conditions
        iterate.conditions = Process()
        iterate.conditions.differentials_a = Methods.Climb.Constant_Throttle_Constant_Body_Angle.update_differentials_altitude
        iterate.conditions.differentials_b = Methods.Common.Numerics.update_differentials_time
        iterate.conditions.velocities      = Methods.Climb.Constant_Throttle_Constant_Body_Angle.update_velocity_vector_from_wind_angle
        iterate.conditions.acceleration    = Methods.Common.Frames.update_acceleration
        iterate.conditions.altitude        = Methods.Common.Aerodynamics.update_altitude
        iterate.conditions.atmosphere      = Methods.Common.Aerodynamics.update_atmosphere
        iterate.conditions.gravity         = Methods.Common.Weights.update_gravity
        iterate.conditions.freestream      = Methods.Common.Aerodynamics.update_freestream
        iterate.conditions.orientations    = Methods.Common.Frames.update_orientations
        iterate.conditions.aerodynamics    = Methods.Common.Aerodynamics.update_aerodynamics
        iterate.conditions.stability       = Methods.Common.Aerodynamics.update_stability
        iterate.conditions.propulsion      = Methods.Common.Energy.update_thrust
        iterate.conditions.weights         = Methods.Common.Weights.update_weights
        iterate.conditions.forces          = Methods.Common.Frames.update_forces
        iterate.conditions.planet_position = Methods.Common.Frames.update_planet_position
        
        # Solve Residuals
        iterate.residuals = Process()
        iterate.residuals.total_forces     = Methods.Climb.Constant_Throttle_Constant_Body_Angle.solve_residuals
        
        # --------------------------------------------------------------
        #   Finalize - after iteration
        # --------------------------------------------------------------
        finalize = self.process.finalize
        
        # Post Processing
        finalize.post_process = Process()        
        finalize.post_process.inertial_position = Methods.Common.Frames.integrate_inertial_horizontal_position
        finalize.post_process.stability         = Methods.Common.Aerodynamics.update_stability
       
        return

