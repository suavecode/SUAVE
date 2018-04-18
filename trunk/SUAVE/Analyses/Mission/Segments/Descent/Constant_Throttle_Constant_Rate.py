## @ingroup Analyses-Mission-Segments-Climb
# Constant_Throttle_Constant_Speed.py
# - ref Constant_Throttle_Constant_Speed
#
# First attempt: 28 Mar 2018, A.A. Wachman

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses.Mission.Segments import Aerodynamic
from SUAVE.Analyses.Mission.Segments import Conditions

from SUAVE.Methods.Missions import Segments as Methods

from SUAVE.Analyses import Process

from SUAVE.Methods.Missions.Segments.Descent import Constant_Throttle_Constant_Rate

# Units
from SUAVE.Core import Units


# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Descent
class Constant_Throttle_Constant_Rate(Aerodynamic):
    """ Descent at a constant throttle setting and a constant rate of descent.
        This segment is being designed to work with zero engine configurations for gliding descents, and uses a desired rate of descent which can be calculated using desired L/D ratio (which can then be used to optimize the configuration.

        Assumptions:
        I know what letters are.
        
        Source:
        N/A
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
        self.altitude_start       = 20. * Units.km # must be greater than end
        self.altitude_end         = 0. * Units.km # default is zero
        self.throttle             = 0.5 # can alter if engines, set engines num to zero if glider
        self.descent_rate         = 3 * Units.m / Units.s # remember z is down
        self.dynamic_pressure     = 150 * Units.Pa # calc this off required L/D and required lift
        

        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
        
        # conditions
        self.state.conditions.update( Conditions.Aerodynamics() )
        
        # initials and unknowns
        ones_row = self.state.ones_row
        self.state.unknowns.body_angle = ones_row(1) * 5.0 * Units.deg
        self.state.unknowns.wind_angle = ones_row(1) * 0.0 * Units.deg
        self.state.residuals.forces    = ones_row(2) * 0.0

        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize - before iteration
        # --------------------------------------------------------------
        initialize = self.process.initialize
        
        initialize.expand_state            = Methods.expand_state
        initialize.differentials           = Methods.Common.Numerics.initialize_differentials_dimensionless
        initialize.conditions              = Methods.Descent.Constant_Throttle_Constant_Rate.initialize_conditions
        initialize.velocities              = Methods.Descent.Constant_Throttle_Constant_Rate.update_velocity_vector_from_wind_angle
        initialize.differentials_altitude  = Methods.Descent.Constant_Throttle_Constant_Rate.update_differentials_altitude      
        
        
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
        iterate.unknowns.mission           = Methods.Descent.Constant_Throttle_Constant_Rate.unpack_body_angle 

        # Update Conditions
        iterate.conditions = Process()
        iterate.conditions.velocities      = Methods.Descent.Constant_Throttle_Constant_Rate.update_velocity_vector_from_wind_angle
        iterate.conditions.differentials_a = Methods.Descent.Constant_Throttle_Constant_Rate.update_differentials_altitude
        iterate.conditions.differentials_b = Methods.Common.Numerics.update_differentials_time
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
        iterate.residuals.total_forces     = Methods.Climb.Common.residual_total_forces
        
        # --------------------------------------------------------------
        #   Finalize - after iteration
        # --------------------------------------------------------------
        finalize = self.process.finalize
        
        # Post Processing
        finalize.post_process = Process()        
        finalize.post_process.inertial_position = Methods.Common.Frames.integrate_inertial_horizontal_position
        finalize.post_process.stability         = Methods.Common.Aerodynamics.update_stability
       
        return

