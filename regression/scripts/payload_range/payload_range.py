
# full_setup.py
#
# Created:  SUAVE Team, Aug 2014
# Modified:

""" setup file for a mission with a E190
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Core import (
Data, Container,
)

# the analysis functions
from plot_mission import plot_mission

from mission_Embraer_E190_constThr_payload_range import full_setup

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
    cruise_segment_tag = "cruise"
    reserves = 1750.
    payload_range_results = payload_range(vehicle,mission,cruise_segment_tag,reserves)    
    
    check_results(payload_range_results)
    
    return


def check_results(new_results):

    return


if __name__ == '__main__':
    main()
