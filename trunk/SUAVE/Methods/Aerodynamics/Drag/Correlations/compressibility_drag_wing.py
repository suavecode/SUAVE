# compressibility_drag_wing.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

def compressibility_drag_wing(conditions,configuration,geometry):
    """ SUAVE.Methods.compressibility_drag_wing(conditions,configuration,geometry)
        computes the induced drag associated with a wing 
        
        Inputs:
        
        Outputs:
        
        Assumptions:
            based on a set of fits
            
    """

    # unpack
    wings      = geometry.Wings
    wing_lifts = conditions.lift_breakdown.clean_wing
    mach       = conditions.mach_number
    
    # start result
    total_compressibility_drag = 0.0
    
    conditions.drag_breakdown.compressibility = Result(total = 0)

    # go go go
    for wing, lift in zip( wings.values(), wing_lifts.values() ):
        
        # unpack wing
        t_c_w   = wing.t_c
        sweep_w = wing.sweep
        cl_w    = lift
    
        # get effective Cl and sweep
        tc = t_c_w /(np.cos(sweep_w))
        cl = cl_w / (np.cos(sweep_w))**2
    
        # compressibility drag based on regressed fits from AA241
        mcc_cos_ws = 0.922321524499352       \
                   - 1.153885166170620*tc    \
                   - 0.304541067183461*cl    \
                   + 0.332881324404729*tc**2 \
                   + 0.467317361111105*tc*cl \
                   + 0.087490431201549*cl**2
            
        # crest-critical mach number, corrected for wing sweep
        mcc = mcc_cos_ws / np.cos(sweep_w)
        
        # divergence mach number
        MDiv = mcc * ( 1.02 + 0.08*(1 - np.cos(sweep_w)) )
        
        # divergence ratio
        mo_mc = mach/mcc
        
        # compressibility correlation, Shevell
        dcdc_cos3g = 0.0019*mo_mc**14.641
        
        # compressibility drag
        cd_c = dcdc_cos3g * (np.cos(sweep_w))**3
        
        # increment
        total_compressibility_drag += cd_c
        
        # dump data to conditions
        wing_results = Result(
            compressibility_drag      = cd_c    ,
            thickness_to_chord        = tc      , 
            wing_sweep                = sweep_w , 
            crest_critical            = mcc     ,
            divergence_mach           = Mdiv    ,
        )
        conditions.drag_breakdown.compressible[wing.tag] = wing_results

    #: for each wing
    
    # dump total comp drag
    conditions.drag_breakdown.compressible.total = total_compressibility_drag
    

    return total_compressibility_drag