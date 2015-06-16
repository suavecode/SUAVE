# full_setup.py
#
# Created:  SUave Team, Aug 2014
# Modified:

""" setup file for a mission with a E190
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

from test_mission_Embraer_E190_constThr import full_setup

from SUAVE.Methods.Performance  import payload_range

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():

    # define the problem
    configs, analyses = full_setup()
    
    configs.finalize()
    analyses.finalize()
    
    vehicle = configs.base
    mission = analyses.missions
    
    # run payload diagram
    cruise_segment_tag = "Cruise"
    reserves = 1750.
    payload_range_results = payload_range(vehicle,mission,cruise_segment_tag,reserves)    
    
    check_results(payload_range_results)
    
    return


def check_results(new_results):

    return


if __name__ == '__main__':
    main()
    plt.show()
