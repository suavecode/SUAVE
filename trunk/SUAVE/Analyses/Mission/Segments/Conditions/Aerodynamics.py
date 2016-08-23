# Aerodynamics.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from Basic import Basic
from Conditions import Conditions


# ----------------------------------------------------------------------
#  Conditions
# ----------------------------------------------------------------------

class Aerodynamics(Basic):
    
    def __defaults__(self):
        self.tag = 'aerodynamic_conditions'
        
        # start default row vectors
        ones_1col = self.ones_row(1)
        ones_2col = self.ones_row(2)
        ones_3col = self.ones_row(3)

        # wind frame conditions
        self.frames.wind = Conditions()
        self.frames.wind.body_rotations        = ones_3col * 0 # rotations in [X,Y,Z] -> [phi,theta,psi]
        self.frames.wind.velocity_vector       = ones_3col * 0
        self.frames.wind.lift_force_vector     = ones_3col * 0
        self.frames.wind.drag_force_vector     = ones_3col * 0
        self.frames.wind.transform_to_inertial = np.empty([0,0,0])

        # body frame conditions
        self.frames.body.thrust_force_vector = ones_3col * 0

        # planet frame conditions
        self.frames.planet = Conditions()
        self.frames.planet.start_time      = None
        self.frames.planet.latitude        = ones_1col * 0
        self.frames.planet.longitude       = ones_1col * 0

        # freestream conditions
        self.freestream = Conditions()        
        self.freestream.velocity           = ones_1col * 0
        self.freestream.mach_number        = ones_1col * 0
        self.freestream.pressure           = ones_1col * 0
        self.freestream.temperature        = ones_1col * 0
        self.freestream.density            = ones_1col * 0
        self.freestream.speed_of_sound     = ones_1col * 0
        self.freestream.dynamic_viscosity  = ones_1col * 0
        self.freestream.altitude           = ones_1col * 0
        self.freestream.gravity            = ones_1col * 0
        self.freestream.reynolds_number    = ones_1col * 0
        self.freestream.dynamic_pressure   = ones_1col * 0

        # aerodynamics conditions
        self.aerodynamics = Conditions()        
        self.aerodynamics.angle_of_attack  = ones_1col * 0
        self.aerodynamics.side_slip_angle  = ones_1col * 0
        self.aerodynamics.roll_angle       = ones_1col * 0
        self.aerodynamics.lift_coefficient = ones_1col * 0
        self.aerodynamics.drag_coefficient = ones_1col * 0
        self.aerodynamics.lift_breakdown              = Conditions()
        self.aerodynamics.drag_breakdown              = Conditions()
        self.aerodynamics.drag_breakdown.parasite     = Conditions()
        self.aerodynamics.drag_breakdown.compressible = Conditions()

        # stability conditions
        self.stability         = Conditions()        
        self.stability.static  = Conditions()
        self.stability.dynamic = Conditions()

        # propulsion conditions
        self.propulsion = Conditions()
        self.propulsion.throttle           = ones_1col * 0
        self.propulsion.battery_energy     = ones_1col * 0
        self.propulsion.battery_voltage    = ones_1col * 0
        self.propulsion.thrust_breakdown       = Conditions()
        self.propulsion.acoustic_outputs       = Conditions()
        self.propulsion.acoustic_outputs.fan   = Conditions()
        self.propulsion.acoustic_outputs.core  = Conditions()

        # energy conditions
        self.energies.gravity_energy       = ones_1col * 0
        self.energies.propulsion_power     = ones_1col * 0
        
        # weights conditions
        self.weights.vehicle_mass_rate     = ones_1col * 0
