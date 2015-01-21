
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses.New_Segment import Simple
from SUAVE.Analyses.New_Segment import Conditions

from SUAVE.Methods.Missions import Segments as Methods

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
        #   Iterate
        # --------------------------------------------------------------
        iterate = self.process.iterate
                
        # Update Initials
        iterate.initials.weights           = Methods.Common.Weights.initialize_weights
        iterate.initials.inertial_position = Methods.Common.Frames.initialize_inertial_position
        iterate.initials.planet_position   = Methods.Common.Frames.initialize_planet_position
        
        # Update Conditions
        iterate.conditions.atmosphere      = Methods.Common.Aerodynamics.update_atmosphere
        iterate.conditions.gravity         = Methods.Common.Weights.update_gravity
        iterate.conditions.freestream      = Methods.Common.Aerodynamics.update_freestream
        iterate.conditions.orientations    = Methods.Common.Frames.update_orientations
        iterate.conditions.aerodynamics    = Methods.Common.Aerodynamics.update_aerodynamics
        iterate.conditions.propulsion      = Methods.Common.Propulsion.update_propulsion
        iterate.conditions.weights         = Methods.Common.Weights.update_weights
        iterate.conditions.forces          = Methods.Common.Frames.update_forces
        iterate.conditions.planet_position = Methods.Common.Frames.update_planet_position


        # --------------------------------------------------------------
        #   Finalize
        # --------------------------------------------------------------
        finalize = self.process.finalize
        finalize.post_process.inertial_position = Methods.Common.Frames.integrate_inertial_position
        finalize.post_process.stability         = Methods.Common.Aerodynamics.update_stability
        
        
        return

