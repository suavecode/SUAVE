# Tim Momose, January 2015
# Modified February 6, 2015

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import pylab as plt
import numpy as np
from copy import deepcopy

import sys

# SUAVE Imports
from SUAVE.Core        import Units
from full_setup_737800 import vehicle_setup
from SUAVE.Analyses.Aerodynamics import AVL


def main():

    # -------------------------------------------------------------
    #  Test Script
    # -------------------------------------------------------------

    # Time the process
    import time
    t0 = time.time()
    print "Start: " + time.ctime()


    tf = time.time()
    print "End:   " + time.ctime()
    print "({0:.2f} seconds)".format(tf-t0)

    plt.show()

    return


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    plt.show()