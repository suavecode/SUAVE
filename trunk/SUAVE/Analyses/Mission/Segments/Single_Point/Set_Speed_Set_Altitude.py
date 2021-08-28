## @ingroup Analyses-Mission-Segments-Single_Point
# Set_Speed_Set_Altitude.py
#
# Created:  Mar 2017, T. MacDonald
# Modified: Jul 2017, T. MacDonald
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
import numpy as np

# Units
from SUAVE.Core import Units


# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Single_Point
class Set_Speed_Set_Altitude(Aerodynamic):
    """ This is a segment that is solved using a single point. A snapshot in time.
        We fix the speed and altitude. Throttle is solved from those.
    
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
        self.altitude  = None
        self.air_speed = 10. * Units['km/hr']
        self.distance  = 10. * Units.km
        self.x_accel   = 0.
        self.z_accel   = 0. # note that down is positive
        self.state.numerics.number_control_points = 1
        
        
        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
        
        # conditions
        self.state.conditions.update( Conditions.Aerodynamics() )
        
        # initials and unknowns
        self.state.unknowns.throttle   = np.array([[0.5]])
        self.state.unknowns.body_angle = np.array([[0.0]])
        self.state.residuals.forces    = np.array([[0.0,0.0]])
        
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize - before iteration
        # --------------------------------------------------------------
        initialize = self.process.initialize
        
        initialize.expand_state            = skip
        initialize.differentials           = skip
        initialize.conditions              = Methods.Single_Point.Set_Speed_Set_Altitude.initialize_conditions

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
        iterate.unknowns.mission           = Methods.Cruise.Common.unpack_unknowns
        
        # Update Conditions
        iterate.conditions = Process()
        iterate.conditions.differentials   = skip
        iterate.conditions.altitude        = Methods.Common.Aerodynamics.update_altitude
        iterate.conditions.atmosphere      = Methods.Common.Aerodynamics.update_atmosphere
        iterate.conditions.gravity         = Methods.Common.Weights.update_gravity
        iterate.conditions.freestream      = Methods.Common.Aerodynamics.update_freestream
        iterate.conditions.orientations    = Methods.Common.Frames.update_orientations
        iterate.conditions.propulsion      = Methods.Common.Energy.update_thrust
        iterate.conditions.aerodynamics    = Methods.Common.Aerodynamics.update_aerodynamics
        iterate.conditions.stability       = Methods.Common.Aerodynamics.update_stability
        iterate.conditions.weights         = Methods.Single_Point.Set_Speed_Set_Altitude.update_weights
        iterate.conditions.forces          = Methods.Common.Frames.update_forces
        iterate.conditions.planet_position = skip

        # Solve Residuals
        iterate.residuals = Process()     
        iterate.residuals.total_forces     = Methods.Climb.Common.residual_total_forces
        
        # --------------------------------------------------------------
        #   Finalize - after iteration
        # --------------------------------------------------------------
        finalize = self.process.finalize
        
        # Post Processing
        finalize.post_process = Process()        
        finalize.post_process.inertial_position = skip
        finalize.post_process.stability         = Methods.Common.Aerodynamics.update_stability
        finalize.post_process.aero_derivatives  = skip
        finalize.post_process.noise             = Methods.Common.Noise.compute_noise
        
        return

