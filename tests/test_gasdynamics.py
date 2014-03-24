""" test_gasdynamics.py: test 1D gasdynamic functions """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import sys
sys.path.append('../trunk')
import SUAVE
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    
    A_A_star = 2.0          # A/A*
    gamma = 1.40            # cp/cv

    # get Mach number
    M = SUAVE.Methods.Gas_Dynamics.mach_from_area_ratio(gamma,A_A_star)
    print "M = ", M

    # recover original area ratio
    A_ratio = SUAVE.Methods.Gas_Dynamics.area_ratio_from_mach(gamma,M)
    print "A/A* = ", A_ratio



    return

# call main
if __name__ == '__main__':
    main()

