# compressible_mixed_flat_plate.py
# 
# Created:  Tim MacDonald, 8/1/14
# Modified:         
# Adapted from compressible_turbulent_flat_plate.py


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import os, sys, shutil
from copy import deepcopy
#from warnings import warn

# package imports
import numpy as np


# ----------------------------------------------------------------------
#  Simple Method
# ----------------------------------------------------------------------


def compressible_mixed_flat_plate(Re,Ma,Tc,xt):

    # Add catching warnings here
    if xt == 0.0:
        cf_inc = 0.0742/(Re)**0.2
    elif xt == 1.0:
        cf_inc = 1.328*xt/(xt*Re)**0.5
    elif xt > 0.0 and xt < 1.0:
        cf_inc = 0.0742/(Re)**0.2 + 1.328*xt/(xt*Re)**0.5 - 0.07425*xt/(xt*Re)**0.2
    else:
        raise ValueError("Turbulent transition must be between 0 and 1")
        
    
    #cf_inc = 0.455/(np.log10(Re))**2.58 * (1.0-xt) + 1.328/(Re)**0.5 * xt
    #print cf_inc
    
    # compressibility correction
    Tw = Tc * (1. + 0.178*Ma**2.)
    Td = Tc * (1. + 0.035*Ma**2. + 0.45*(Tw/Tc - 1.))
    k_comp = (Tc/Td) 
    
    # reynolds correction
    Rd_w = Re * (Td/Tc)**1.5 * ( (Td+216.) / (Tc+216.) )
    k_reyn = (Re/Rd_w)**0.2
    
    # apply corrections
    cf_comp = cf_inc * k_comp * k_reyn
    
    return cf_comp, k_comp, k_reyn

  
# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
if __name__ == '__main__':    
    
    (cf_comp, k_comp, k_reyn) = compressible_mixed_flat_plate(1.0*10.0**8.0,0.0,216.0,0.0)
    
    print cf_comp
    print k_comp
    print k_reyn
