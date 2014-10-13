""" cruise_range.py: ... """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import math
import copy

from SUAVE.Structure            import Data
from SUAVE.Attributes.Results   import Result, Segment
# from SUAVE.Methods.Utilities    import chebyshev_data, pseudospectral

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def cruise_range(vehicle,mission,maxto,fuel_burn_maneuever,fuel_burn_climb_inc,reserve_fuel,sfc_sfcref,sls_thrust,eng_type,mzfw_ratio):
    """Calculate the range from the initial and final weight, uses the Breguet range equation to calculate the range after the segment. Output is in kilometers
        
        Assumptions:
            Steady level flight so lift = weight and thrust = drag
            The Mach number and speed of sound were average values based on the prescribed initial and final values specified 
        
        Inputs:
            mzfw_ratio = vehicle.Mass.fmzfw
            mission.segments['Initial Cruise'].atmo
            mission.segments['Initial Cruise'].alt
            mission.segments['Final Cruise'].alt
        
        Outputs
            range_cruise
            fuel_cruise
        
    """
    # AA 241 Notes Section 11.4
    
    # Atmosphere
    atmo = mission.segments['Initial Cruise'].atmo
    atm = Atmosphere()
    atm.dt = 0.0  #Return ISA+(value)C properties
    
    # Initial Cruise Atmospheric Conditions
    atm.alt  = mission.segments['Initial Cruise'].alt # in km
    Climate.EarthIntlStandardAtmosphere(atm)
    speed_of_sound_in = atm.vsnd
    
    # Final Cruise Atmospheric Conditions
    atm.alt = mission.segments['Final Cruise'].alt
    Climate.EarthIntlStandardAtmosphere(atm)
    speed_of_sound_fin = atm.vsnd
    
    # Unpack
    mach_cr_initial = mission.segments['Initial Curise'].mach
    mach_cr_final = mission.segments['FInal Cruise'].mach
    
    # Calculate
    mzfw = maxto * mzfw_ratio
    weight_initial = maxto - fuel_burn_maneuever - fuel_burn_climb_inc
    weight_final = mzfw + fuel_burn_maneuever + reserve_fuel
    lift = (weight_final + weight_initial) / 2 # Assume Steady Level Flight
    mach_cr_av = (mach_cr_final + mach_cr_initial) / 2 # Average Mach number
    speed_of_sound_av = (speed_of_sound_fin +speed_of_sound_in) / 2 # Average speed of sound
    sfc,thrust = Propulsion.engine_analysis( mach_cr_average, speed_of_sound_av, sfc_sfcref,sls_thrust,eng_type)
    drag = thrust # Assume steady level flight
    cruise_velocity = (mach_cr_final * speed_of_sound_fin + mach_cr_initial *speed_of_sound_in) / 2 * 3600 / 1000 # averaged velocities at initial and final cruise
    range_cruise = cruise_velocity / sfc * lift / drag * math.log( weight_initial /weight_final) #in km
    fuel_cruise = weight_initial - weight_final
    
    return (range_cruise,fuel_cruise)
