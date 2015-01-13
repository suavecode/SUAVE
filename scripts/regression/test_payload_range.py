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

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Core import (
Data, Container, Data_Exception, Data_Warning,
)

# the analysis functions
from the_aircraft_function import the_aircraft_function
from plot_mission import plot_mission

from test_mission_Embraer_E190_constThr import vehicle_setup, mission_setup

from SUAVE.Methods.Performance  import payload_range


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():

    # define the problem
    vehicle, mission = full_setup()
    
    # run payload diagram
    cruise_segment_tag = "Cruise"
    payload_range_results = payload_range(vehicle,mission,cruise_segment_tag)    
    
    check_results(payload_range_results)
    
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    vehicle = vehicle_setup() # imported from E190 test script
    mission = mission_setup(vehicle)

    return vehicle, mission


def check_results(new_results):

    return


if __name__ == '__main__':
    main()
    plt.show()