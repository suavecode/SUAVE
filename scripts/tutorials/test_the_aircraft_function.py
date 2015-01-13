
# test_the_aircraft_function.py
# 
# Created:  Trent Lukaczyk , Aug 2014
# Modified: 


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Attributes import Units
from SUAVE.Core import Data

import numpy as np
import pylab as plt

import copy, time

from full_setup            import full_setup
from full_setup_737800     import full_setup_737800
from full_setup_AS2        import full_setup_AS2
from the_aircraft_function import the_aircraft_function
from post_process          import post_process


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    vehicle, mission = full_setup_AS2()
    
    results = the_aircraft_function(vehicle,mission)
    
    post_process(vehicle,mission,results)
    
    return


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()
    plt.show(block=True) # here so as to not block the regression test