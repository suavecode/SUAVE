""" weight_climb_added.py: ... """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import math
import SUAVE.Methods.Units
import copy

from SUAVE.Structure            import Data
from SUAVE.Attributes.Results   import Result, Segment
# from SUAVE.Methods.Utilities    import chebyshev_data, pseudospectral

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def weight_climb_added(vehicle,mission,maxto):
    """Fuel increment added to cruise fuel burned over distance when aircraft is climbing to the cruise altitude
        
        Assumptions:
        
        Inputs:
            mission.segments['Initial Cruise'].mach
            mission.segments['Initial Cruise'].atmo
            mission.segments['Initial Cruise'].alt
        
        Outputs:
            fuel_burn_climb_inc
        
     """
    # AA 241 Notes Section 11.3
    
    # Atmosphere
    atmo = mission.segments['Initial Cruise'].atmo
    atm = Atmosphere()
    atm.alt  = mission.segments['Initial Cruise'].alt # in km
    atm.dt = 0.0  #Return ISA+(value)C properties
    Climate.EarthIntlStandardAtmosphere(atm)
    
    # Unpack
    cruise_mach = mission.segments['Initial Cruise'].mach
    speed_of_sound = atm.vsnd #in m/s
    
    # Calculate
    cruise_speed = speed_of_sound * cruise_mach / 1000 * 3600 # in km/hr
    cruise_velocity = Units.ConvertDistance(cruise_speed,'km','nm') #Converting to knots
    alt_kft = Units.ConvertLength( alt * 1000 ,'m','ft') / 1000 #Converting to kft
    fuel_burn_climb_inc = maxto * (alt_kft / 31.6 + (cruise_velocity / 844) ** 2)
    
    return fuel_burn_climb_inc