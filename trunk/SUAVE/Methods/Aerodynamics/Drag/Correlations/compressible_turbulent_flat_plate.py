# compressible_turbulent_flat_plate.py
# 
# Created:  Your Name, Dec 2013
# Modified:         


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np


# ----------------------------------------------------------------------
#  Simple Method
# ----------------------------------------------------------------------


def compressible_turbulent_flat_plate(Re,Ma,T):
    
    # incompressible skin friction coefficient
    cf_inc = 0.455/(np.log10(Re_w))**2.58
    
    # compressibility correction
    Tw = Tc * (1. + 0.178*Mc**2.)
    Td = Tc * (1. + 0.035*Mc**2. + 0.45*(Tw/Tc - 1.))
    k_comp = (Tc/Td) 
    
    # reynolds correction
    Rd_w = Re_w * (Td/Tc)**1.5 * ( (Td+216.) / (Tc+216.) )
    k_reyn = (Re_w/Rd_w)**0.2
    
    # apply corrections
    cf_comp = cf_inc * k_comp * k_reyn
    
    return cf_comp, k_comp, k_reyn

  
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':    
    raise RuntimeError , 'test failed, not implemented'