# Ground.py
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
from SUAVE.Core import Data


# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Ground(Aerodynamic):
    """
        Base segment for takeoff and landing segments. Integrates equations of motion
        including rolling friction.
        Notes Regarding Friction Coefficients
        Dry asphalt or concrete: .04 brakes off, .4 brakes on
        Wet asphalt or concrete: .05 brakes off, .225 brakes on
        Icy asphalt or concrete: .02 brakes off, .08 brakes on
        Hard turf:               .05 brakes off, .4 brakes on
        Firm dirt:               .04 brakes off, .3 brakes on
        Soft turf:               .07 brakes off, .2 brakes on
        Wet grass:               .08 brakes off, .2 brakes on
        FROM: General Aviation Aircraft Design: Applied Methods and Procedures,
        by Snorri Gudmundsson, copyright 2014, published by Elsevier, Waltham,
        MA, USA [p.938]
    """

    # ------------------------------------------------------------------
    #   Data Defaults
    # ------------------------------------------------------------------  

    def __defaults__(self):
        
        # --------------------------------------------------------------
        #   User inputs
        # --------------------------------------------------------------
        self.ground_incline       = 0.0
        self.friction_coefficient = 0.04
        self.throttle             = None
        self.velocity_start       = 0.0
        self.velocity_end         = 0.0 
        
        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
    
        # conditions
        self.state.conditions.update( Conditions.Aerodynamics() )
    
        # initials and unknowns
        ones_row = self.state.ones_row
        self.state.unknowns.velocity_x            = ones_row(1) * 0.0
        self.state.unknowns.time                  = 0.1
        self.state.residuals.acceleration_x       = ones_row(1) * 0.0
        self.state.residuals.final_velocity_error = ones_row(1) * 0.0
    
        # Specific ground things
        self.state.conditions.ground = Data()
        self.state.conditions.ground.incline                      = ones_row(1) * 0.0
        self.state.conditions.ground.friction_coefficient         = ones_row(1) * 0.0
        self.state.conditions.frames.inertial.ground_force_vector = ones_row(3) * 0.0
    
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
    
        # --------------------------------------------------------------
        #   Initialize - before iteration
        # --------------------------------------------------------------
        initialize = self.process.initialize
    
        initialize.expand_state            = Methods.expand_state
        initialize.differentials           = Methods.Common.Numerics.initialize_differentials_dimensionless
        initialize.conditions              = Methods.Ground.Common.initialize_conditions        
    
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
        iterate.unknowns.mission           = Methods.Ground.Common.unpack_unknowns
    
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
        iterate.conditions.forces_ground   = Methods.Ground.Common.compute_ground_forces
        iterate.conditions.forces          = Methods.Ground.Common.compute_forces
        iterate.conditions.planet_position = Methods.Common.Frames.update_planet_position
        
        # Solve Residuals
        iterate.residuals = Process()     
        iterate.residuals.total_forces     = Methods.Ground.Common.solve_residuals
    
        # --------------------------------------------------------------
        #   Finalize - after iteration
        # --------------------------------------------------------------
        finalize = self.process.finalize
    
        # Post Processing
        finalize.post_process = Process()        
        finalize.post_process.inertial_position = Methods.Common.Frames.integrate_inertial_horizontal_position
        finalize.post_process.stability         = Methods.Common.Aerodynamics.update_stability  
        finalize.post_process.ground            = Methods.Ground.Common.post_process

        return