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

def find_takeoff_weight_given_tofl(vehicle,takeoff_config,airport,target_tofl):
    """ SUAVE.Methods.Perfomance.find_takeoff_weight_given_tofl(vehicle,takeoff_config,airport,target_tofl)
        This routine estimates the takeoff weight given a certain takeoff field lenght

        Inputs:
            vehicle - SUAVE type vehicle

            takeoff_config - data dictionary containing:
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

    tow_lower = takeoff_config.mass_properties.operating_empty
    tow_upper = 1.10 * takeoff_config.mass_properties.max_takeoff

#saving initial reference takeoff weight
    tow_ref = takeoff_config.mass_properties.max_takeoff

    tow_vec = np.linspace(tow_lower,tow_upper,50.)
    tofl = np.zeros_like(tow_vec)

    for id,tow in enumerate(tow_vec):
        takeoff_config.mass_properties.takeoff = tow
        tofl[id] = estimate_take_off_field_length(vehicle,takeoff_config,airport)

    max_tow = np.interp(target_tofl,tofl,tow_vec)

#reset the initial takeoff weight
    takeoff_config.mass_properties.max_takeoff = tow_ref

    return max_tow


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':
    raise RuntimeError , 'test failed, not implemented'