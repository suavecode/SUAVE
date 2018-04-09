## @ingroup Analyses-Mission-Segments-Climb
# Unknown_Throttle.py
#
# Created:  Feb 2018, E. Botero, W. Maier
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

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------
## @ingroup Analyses-Mission-Segments-Climb
class Linear_Throttle_Fixed_Angle_Time(Aerodynamic):
    """ This is a basic climb segment that is not callable by a user.
       
        Assumptions:
        None
        
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
        self.altitude_start    = None 
        self.air_speed_start   = 00. * Units.m / Units.s
        self.flight_path_angle = 90. * Units.degrees
        self.flight_time       = 1.0 * Units.seconds
        self.throttle_start    = 1.0
        self.throttle_end      = 1.0
        
        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
        
        # Conditions
        self.state.conditions.update( Conditions.Aerodynamics() )
        
        # Initials and unknowns
        ones_row                       = self.state.ones_row
        self.state.unknowns.body_angle = ones_row(1) * 89.0 * Units.degrees
        self.state.unknowns.air_speed  = ones_row(1) * 100.0 * Units.m / Units.s
        self.state.residuals.forces    = ones_row(2) * 0.0
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize - before iteration
        # --------------------------------------------------------------
        initialize                         = self.process.initialize
        initialize.expand_state            = Methods.expand_state
        initialize.differentials           = Methods.Common.Numerics.initialize_differentials_dimensionless
        initialize.conditions              = Methods.Climb.Linear_Throttle_Fixed_Angle_Time.initialize_conditions
        initialize.differentials_time      = Methods.Common.Numerics.update_differentials_time 
        
        # --------------------------------------------------------------
        #   Converge - starts iteration
        # --------------------------------------------------------------
        converge                           = self.process.converge
        converge.converge_root             = Methods.converge_root        
        
        # --------------------------------------------------------------
        #   Iterate - this is iterated
        # --------------------------------------------------------------
        iterate                            = self.process.iterate
                
        # Update Initials
        iterate.initials                   = Process()
        iterate.initials.time              = Methods.Common.Frames.initialize_time
        iterate.initials.weights           = Methods.Common.Weights.initialize_weights
        iterate.initials.inertial_position = Methods.Common.Frames.initialize_inertial_position
        iterate.initials.planet_position   = Methods.Common.Frames.initialize_planet_position
        
        # Unpack Unknowns
        iterate.unknowns                   = Process()
        iterate.unknowns.mission           = Methods.Climb.Linear_Throttle_Fixed_Angle_Time.unpack_unknowns
        
        # Update Conditions
        iterate.conditions                 = Process()
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
        iterate.residuals                  = Process()
        iterate.residuals.total_forces     = Methods.Climb.Common.residual_total_forces
        
        # --------------------------------------------------------------
        #   Finalize - after iteration
        # --------------------------------------------------------------
        finalize = self.process.finalize
        
        # Post Processing
        finalize.post_process                   = Process()        
        finalize.post_process.inertial_position = Methods.Common.Frames.integrate_inertial_horizontal_position
        finalize.post_process.stability         = Methods.Common.Aerodynamics.update_stability
       
        return

