
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from SUAVE.Core import Data, Data_Exception
from SUAVE.Analyses import Process
from SUAVE.Analyses.New_Segment import Segment, State
from SUAVE.Analyses.New_Segment import Conditions

from SUAVE.Methods.Missions import Segments as SegMethods


# Units
from SUAVE.Core import Units


# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

class Cruise(Segment):
    
    def __defaults__(self):
        
        # --------------------------------------------------------------
        #  User inputs
        # --------------------------------------------------------------
        self.altitude  = 10. * Units.km
        self.air_speed = 10. * Units['km/hr']
        self.distance  = 10. * Units.km
        
        
        # --------------------------------------------------------------
        #  State
        # --------------------------------------------------------------
        
        # conditions
        self.state.conditions.update( Conditions.Aerodynamics() )
        
        # initials and unknowns
        ones_row = self.state.ones_row
        self.state.unknowns.throttle   = ones_row(1) * 0.5
        self.state.unknowns.body_angle = ones_row(1) * 0.0
        self.state.residuals.forces    = ones_row(2) * 0.0
        
        
        # --------------------------------------------------------------
        #  Process
        # --------------------------------------------------------------
        
        # initialize
        initialize = self.process.initialize
        initialize.expand_state  = SegMethods.expand_state
        initialize.differentials = SegMethods.Common.Numerics.initialize_differentials_dimensionless
        initialize.conditions    = SegMethods.Cruise.Common.initialize_conditions
        
        # converge
        converge = self.process.converge
        converge.converge_root = SegMethods.converge_root

        # iterate
        iterate = self.process.iterate
        
        # unpack unknowns
        iterate.unpack_unknowns = SegMethods.Cruise.Common.unpack_unknowns
        
        # update initials
        iterate.initials = Process()
        iterate.initials.weights           = SegMethods.Common.Weights.initialize_weights
        iterate.initials.inertial_position = SegMethods.Common.Frames.initialize_inertial_position
        iterate.initials.time              = SegMethods.Common.Frames.initialize_time
        iterate.initials.battery           = SegMethods.Common.Energy.initialize_battery
        iterate.initials.planet_position   = SegMethods.Common.Frames.initialize_planet_position
        
        # update conditions
        iterate.conditions = Process()
        iterate.conditions.differentials   = SegMethods.Common.Numerics.update_differentials_time
        iterate.conditions.atmosphere      = SegMethods.Common.Aerodynamics.update_atmosphere
        iterate.conditions.gravity         = SegMethods.Common.Weights.update_gravity
        iterate.conditions.freestream      = SegMethods.Common.Aerodynamics.update_freestream
        iterate.conditions.planet_position = SegMethods.Common.Frames.update_planet_position
        iterate.conditions.orientations    = SegMethods.Common.Frames.update_orientations
        iterate.conditions.aerodynamics    = SegMethods.Common.Aerodynamics.update_aerodynamics
        iterate.conditions.propulsion      = SegMethods.Common.Propulsion.update_propulsion
        iterate.conditions.weights         = SegMethods.Common.Weights.update_weights
        iterate.conditions.forces          = SegMethods.Common.Frames.update_forces
        
        # solve residuals
        iterate.residuals = Process()
        iterate.residuals.total_forces     = SegMethods.Cruise.Common.residual_total_forces

        # finalize
        finalize = self.process.finalize
        finalize.post_process = Process()
        finalize.post_process.position  = SegMethods.Common.Frames.integrate_inertial_position
        finalize.post_process.stability = SegMethods.Common.Aerodynamics.update_stability
        
        
        return

    