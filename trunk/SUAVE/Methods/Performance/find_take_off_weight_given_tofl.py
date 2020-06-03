## @ingroup Methods-Performance
# find_take_off_weight_given_tofl.py
#
# Created:  Sep 2014, C. Ilario, T. Orra 
# Modified: Jan 2016, E. Botero


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Performance.estimate_take_off_field_length import estimate_take_off_field_length

import numpy as np

# ----------------------------------------------------------------------
#  Find Takeoff Weight Given TOFL
# ----------------------------------------------------------------------

## @ingroup Methods-Performance
def find_take_off_weight_given_tofl(vehicle,analyses,airport,target_tofl):
    """Estimates the takeoff weight given a certain takeoff field length.

    Assumptions:
    assumptions per estimate_take_off_field_length()

    Source:
    N/A

    Inputs:
    vehicle.mass_properties.
      operating_empty         [kg]
      max_takeoff             [kg]
      analyses                [SUAVE data structure]
      airport                 [SUAVE data structure]
      target_tofl             [m]
      
    Outputs:
    max_tow                   [kg]

    Properties Used:
    N/A
    """       

    #unpack
    tow_lower = vehicle.mass_properties.operating_empty
    tow_upper = 1.10 * vehicle.mass_properties.max_takeoff

    #saving initial reference takeoff weight
    tow_ref = vehicle.mass_properties.max_takeoff

    tow_vec = np.linspace(tow_lower,tow_upper,50)
    tofl    = np.zeros_like(tow_vec)

    for id,tow in enumerate(tow_vec):
        vehicle.mass_properties.takeoff = tow
        tofl[id] = estimate_take_off_field_length(vehicle,analyses,airport)

    target_tofl = np.atleast_1d(target_tofl)
    max_tow = np.zeros_like(target_tofl)

    for id,toflid in enumerate(target_tofl):
        max_tow[id] = np.interp(toflid,tofl,tow_vec)

    #reset the initial takeoff weight
    vehicle.mass_properties.max_takeoff = tow_ref

    return max_tow