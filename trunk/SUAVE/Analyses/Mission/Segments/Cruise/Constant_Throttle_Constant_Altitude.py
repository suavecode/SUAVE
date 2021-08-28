## @ingroup Analyses-Mission-Segments-Cruise
# Constant_Throttle_Constant_Altitude.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff
#           Mar 2020, M. Clarke
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

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Cruise
class Constant_Throttle_Constant_Altitude(Aerodynamic):
    """ Vehicle flies at a set throttle setting. Allows a vehicle to do a level acceleration.
    
        Assumptions:
        None
        
        Source:
        None
    """           
    
    
    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  

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
        self.throttle        = None
        self.altitude        = None
        self.air_speed_start = None
        self.air_speed_end   = 0.0 
        
        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
    
        # conditions
        self.state.conditions.update( Conditions.Aerodynamics() )
    
        # initials and unknowns
        ones_row = self.state.ones_row
        self.state.unknowns.body_angle            = ones_row(1) * 0.0
        self.state.unknowns.velocity_x            = ones_row(1) * 0.0
        self.state.unknowns.time                  = 100.
        self.state.residuals.final_velocity_error = 0.0
        self.state.residuals.forces               = ones_row(2) * 0.0
    
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
    
        # --------------------------------------------------------------
        #   Initialize - before iteration
        # --------------------------------------------------------------
        initialize = self.process.initialize
    
        initialize.expand_state            = Methods.expand_state
        initialize.differentials           = Methods.Common.Numerics.initialize_differentials_dimensionless
        initialize.conditions              = Methods.Cruise.Constant_Throttle_Constant_Altitude.initialize_conditions        
    
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
        iterate.unknowns.mission           = Methods.Cruise.Constant_Throttle_Constant_Altitude.unpack_unknowns        
    
        # Update Conditions
        iterate.conditions = Process()
        iterate.conditions.differentials   = Methods.Common.Numerics.update_differentials_time
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
        iterate.residuals.total_forces     = Methods.Cruise.Constant_Throttle_Constant_Altitude.solve_residuals
    
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