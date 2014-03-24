""" climb_gradient.py: ... """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import math
import SUAVE.Methods.Units
import copy

from SUAVE.Structure            import Data
from SUAVE.Attributes.Results   import Result, Segment

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def climb_gradient(vehicle,mission,maxto,sfc_sfcref,sls_thrust,eng_type):

    """Estimating the climb gradient of the aircraft
        
        Assumptions:
        
        Inputs:
            mission.segments['Initial Cruise'].atmo
            mission.segments['Initial Cruise'].alt
            mission.segments['Take-Off'].mach
            vehicle.Turbo_Fan['TheTurboFan'].pos_dy
            vehicle.Wing['Vertical Tail'].span
        
        Outputs:
            climb_grad
        
        
     """
    # AA 241 Notes Section 11.3
    
    # Atmosphere
    atmo = mission.segments['Initial Cruise'].atmo
    atm = Atmosphere()
    atm.alt  = mission.segments['Initial Cruise'].alt # in km
    atm.dt = 0.0  #Return ISA+(value)C properties
    Climate.EarthIntlStandardAtmosphere(atm)
    
    # Unpack
    mach_to = mission.segments['Take-Off'].mach
    speed_of_sound = atm.vsnd # should be in m/s
    pressure = atm.pressure # ambient static pressure
    rho = atm.density # density at takeoff altitude
    y_engine = vehicle.Turbo_Fan['TheTurboFan'].pos_dy # distance from fuselage centerline to critical engine
    height_vert = vehicle.Wing['Vertical Tail'].span # Make sure this is the right dimension
    
    length_vert = 0 # distance from c.g. to vertical tail a.c. <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    
    # Calculate
    sfc,thrust_single = Propulsion.engine_analysis(mach_to,speed_of_sound,sfc_sfcref,sls_thrust,eng_type) #Critical takeoff thrust
    
    drag_climb = 0 # Need to determine how to calculate the drag in climb configuration <<<
    
    area_inlet = 0 # Need to calculate the area of the engine inlet <<<<<<<<<<<<<<<<<<<<<<<
    
    drag_windmill = 0.0044 * pressure * area_inlet
    vel_climb = mach_to * speed_of_sound # Velocity at takeoff
    pressure_dyn = 0.5 * rho * vel_climb ** 2
    drag_trim = y_engine ** 2 * (thrust_single+drag_windmill) ** 2 / (pressure_dyn * math.pi * height_vert ** 2 * length_vert ** 2)
    drag_engine_out = drag_climb + drag_windmill + drag_trim
    climb_grad = (thrust_single-drag_engine_out) / maxto # at constant speed
    
    return climb_grad