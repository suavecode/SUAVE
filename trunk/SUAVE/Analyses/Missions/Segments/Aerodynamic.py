
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses.Missions.Segments import Simple
from SUAVE.Analyses.Missions.Segments import Conditions

from SUAVE.Methods.Missions import Segments as Methods

from SUAVE.Analyses import Process

# Units
from SUAVE.Core import Units


# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

class Aerodynamic(Simple):
    
    def __defaults__(self):
        
        # --------------------------------------------------------------
        #   User inputs
        # --------------------------------------------------------------
        # self.example = 1.0
        
        
        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
        
        # conditions
        self.state.conditions.update( Conditions.Aerodynamics() )
        
        # initials and unknowns, Example...
        ##ones_row = self.state.ones_row
        ##self.state.unknowns.throttle   = ones_row(1) * 0.5
        ##self.state.unknowns.body_angle = ones_row(1) * 0.0
        ##self.state.residuals.forces    = ones_row(2) * 0.0
        
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize - before iteration
        # --------------------------------------------------------------
        initialize = self.process.initialize
        initialize.clear()
        
        initialize.expand_state            = Methods.expand_state
        initialize.differentials           = Methods.Common.Numerics.initialize_differentials_dimensionless
        initialize.conditions              = None        
        
        # --------------------------------------------------------------
        #   Converge - starts iteration
        # --------------------------------------------------------------
        converge = self.process.converge
        converge.clear()
        
        converge.converge_root             = Methods.converge_root        
        
        # --------------------------------------------------------------
        #   Iterate - this is iterated
        # --------------------------------------------------------------
        iterate = self.process.iterate
        iterate.clear()
                
        # Update Initials
        iterate.initials = Process()
        iterate.initials.time              = Methods.Common.Frames.initialize_time
        iterate.initials.weights           = Methods.Common.Weights.initialize_weights
        iterate.initials.inertial_position = Methods.Common.Frames.initialize_inertial_position
        iterate.initials.planet_position   = Methods.Common.Frames.initialize_planet_position
        
        # Unpack Unknowns
        iterate.unpack_unknowns            = None  
        
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
        iterate.conditions.propulsion      = Methods.Common.Propulsion.update_propulsion
        iterate.conditions.weights         = Methods.Common.Weights.update_weights
        iterate.conditions.forces          = Methods.Common.Frames.update_forces
        iterate.conditions.planet_position = Methods.Common.Frames.update_planet_position

        # Solve Residuals
        iterate.residuals = Process()

        # --------------------------------------------------------------
        #   Finalize - after iteration
        # --------------------------------------------------------------
        finalize = self.process.finalize
        finalize.clear()
        
        # Post Processing
        finalize.post_process = Process()        
        finalize.post_process.inertial_position = Methods.Common.Frames.integrate_inertial_horizontal_position
        finalize.post_process.stability         = Methods.Common.Aerodynamics.update_stability
        
        return

