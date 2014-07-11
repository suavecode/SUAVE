""" estimate_landing_field_length.py: ... """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import math
import Units
import copy

from SUAVE.Structure            import Data
from SUAVE.Attributes.Results   import Result, Segment
from Utilities                  import chebyshev_data, pseudospectral

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def estimate_landing_field_length(vehicle,mission,maxto,mzfw_ratio,fuel_burn_maneuever,reserve_fuel,sref,sfc_sfcref,sls_thrust,eng_type):
    
    """ Calculating the landing distance of an aircraft through use of combined aircraft deceleration at constant altitude and ground deceleration
        
        Assumptions:
            Braking coefficient was assumed to be 0.2 (from http://www.airporttech.tc.faa.gov/naptf/att07/2002%20TRACK%20S.pdf/S10.pdf)
        
        Inputs:
        
        Outputs:
        
        
    """
    # AA 241 Notes Section 11.2
        
    # Atmosphere
    atmo = mission.segments['Landing'].atmo
    atm = Atmosphere()
    atm.alt  = mission.segments['Landing'].alt # in km
    atm.dt = 0.0  #Return ISA+(value)C properties
    Climate.EarthIntlStandardAtmosphere(atm)
        
    # Unpack
    rho = atm.density
    speed_of_sound = atm.vsnd
    flap = mission.segments['Landing'].flap_setting
    slat = mission.segments['Landing'].slat_setting
    mach_land = mission.segments['Landing'].mach
        
    # Calculations
    mzfw = maxto * mzfw_ratio
    weight_landing = mzfw + fuel_burn_maneuever + reserve_fuel
    
    clmax = 0 # Need to calculate the maximum lift coefficient<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    vel_stall = math.sqrt( 2 * weight_landing / (rho * clmax *sref))
    vel_50 = 1.3 * vel_stall
    
    lift_landing = 0 # Lift generated in the landing configuration<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    drag_landing = 0 # Drag in the landing configuration<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    sfc,thrust_engine = Propulsion.engine_analysis(mach_land,speed_of_sound,sfc_sfcref,sls_thrust,eng_type)
    engine_number = len(vehicle.Propulsors.get_TurboFans())
    thrust_landing = thrust_engine * engine_number
    lift_drag_eff = lift_landing / (thrust_landing - drag_landing)
    vel_landing = 1.25 * vel_stall
    landing_ground = 50 * lift_drag_eff + lift_drag_eff * (vel_50 ** 2 - vel_landing ** 2) / 2 / constants.grav
    mu = 0.2 # braking coefficient of friction
    
    lift_ground = 0 #lift in landing configuration on ground<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    drag_ground = 0 #drag in landing configuration on ground<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    resistance = mu * (weight_landing - lift_ground) + drag_ground
    landing_ground = vel_landing ** 2 * weight_landing / 2 / resistance / constants.grav
    lfl = (landing_air + landing_ground) / 0.6
    
    return lfl
