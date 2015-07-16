# find_takeoff_weight_given_tofl.py
#
# Created:  Carlos and Tarik, Sept 2014
# Modified:


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Methods.Performance.estimate_take_off_field_length import estimate_take_off_field_length

# package imports
import numpy as np



# ----------------------------------------------------------------------
#  Simple Method
# ----------------------------------------------------------------------

def find_takeoff_weight_given_tofl(vehicle,analyses,airport,target_tofl):
    """ SUAVE.Methods.Perfomance.find_takeoff_weight_given_tofl(vehicle,takeoff_config,airport,target_tofl)
        This routine estimates the takeoff weight given a certain takeoff field lenght

        Inputs:
            analyses - ? ?  ? 

            vehicle - data dictionary containing:
                 mass_properties.operating_empty
                 mass_properties.max_takeoff

            airport   - SUAVE type airport data, with followig fields:
                atmosphere                  - Airport atmosphere (SUAVE type)
                altitude                    - Airport altitude
                delta_isa                   - ISA Temperature deviation

            target_tofl - The available field lenght for takeoff

        Outputs:
            max_tow - Maximum takeoff weight for a given field lenght

    """

#unpack

    tow_lower = vehicle.mass_properties.operating_empty
    tow_upper = 1.10 * vehicle.mass_properties.max_takeoff

#saving initial reference takeoff weight
    tow_ref = vehicle.mass_properties.max_takeoff

    tow_vec = np.linspace(tow_lower,tow_upper,50.)
    tofl = np.zeros_like(tow_vec)

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