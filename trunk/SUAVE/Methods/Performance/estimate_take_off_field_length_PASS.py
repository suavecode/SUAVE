""" estimate_take_off_field_length.py: ... """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import math
import copy

from SUAVE.Core            import Data
from SUAVE.Attributes.Results   import Result, Segment
# from SUAVE.Methods.Utilities    import chebyshev_data, pseudospectral

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def estimate_take_off_field_length(vehicle,mission,maxto,sref,sfc_sfcref,sls_thrust,eng_type):
    """ Calculating the TOFL based on a parametric fit from the AA 241 Notes
        
        Assumptions:
        
        Inputs:
            mission.segments['Take-Off'].alt
            mission.segments['Take-Off'].flap_setting
            mission.segments['Take-Off'].slat_setting
            mission.segments['Take-Off'].mach
            mission.segments['Take-Off'].atmo
        
        Outputs:
            tofl
        
    """
            
    # AA 241 Notes Section 11.1
    
    # Atmosphere
    atmo = mission.segments['Take-Off'].atmo # Not needed while only have ISA
    atm = Atmosphere()
    atm.alt = mission.segments['Take-Off'].alt # Needs to be in meters
    atm.dt = 0.0  #Return ISA+(value)C properties
    Climate.EarthIntlStandardAtmosphere(atm)
    
    # Unpack
    rho_standard = constants.rho0 #density at standard sea level temperature and pressure
    rho = atm.density # density at takeoff altitude
    speed_of_sound = atm.vsnd # should be in m/s
    flap = mission.segments['Take-Off'].flap_setting
    slat = mission.segments['Take-Off'].slat_setting
    mach_to = mission.segments['Take-Off'].mach
    
    # Calculate
        
    clmax = 0 # Need way to calculate in Aerodynamics<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    sfc,thrust_takeoff = Propulsion.engine_analysis( 0.7 * mach_to, speed_of_sound,sfc_sfcref,sls_thrust,eng_type)  #use 70 percent of takeoff velocity
    sigma = rho / rho_standard
    engine_number = len(vehicle.Propulsors.get_TurboFans())
    index = maxto ** 2 / ( sigma * clmax * sref * (engine_number * thrust_takeoff) )
    if engine_number == 2:
        tofl = 857.4 + 28.43 * index + 0.0185 * index ** 2
    elif engine_number == 3:
        tofl = 667.9 + 26.91 * index + 0.0123 * index ** 2
    elif engine_number == 4:
        tofl = 486.7 + 26.20 * index + 0.0093 * index ** 2
    else:
        tofl = 486.7 + 26.20 * index + 0.0093 * index ** 2
        print 'Incorrect number of engines for TOFL' #Error Message for <2 or >4 engines
    
    return tofl
