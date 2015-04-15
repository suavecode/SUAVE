# compressibility_drag_wing.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Core import Results
from SUAVE.Core import (
    Data, Container, Data_Exception, Data_Warning,
)

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

def compressibility_drag_wing(state,settings,geometry):
#def compressibility_drag_wing(conditions,configuration,geometry):
    """ SUAVE.Methods.compressibility_drag_wing(conditions,configuration,geometry)
        computes the induced drag associated with a wing 
        
        Inputs:
        
        Outputs:
        
        Assumptions:
            based on a set of fits
            
    """

    # unpack
    conditions = state.conditions
    configuration = settings
    
    
    wing      = geometry
    wing_lifts = conditions.aerodynamics.lift_breakdown.compressible_wings # currently the total aircraft lift
    mach       = conditions.freestream.mach_number
    drag_breakdown = conditions.aerodynamics.drag_breakdown
    
    # start result
    total_compressibility_drag = 0.0
    #drag_breakdown.compressible = Results()

    
        
    # unpack wing
    t_c_w   = wing.thickness_to_chord
    sweep_w = wing.sweep
    
    # Currently uses vortex lattice model on all wings
    if wing.tag=='main_wing':
        cl_w = wing_lifts
    else:
        cl_w = 0

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
    #total_compressibility_drag += cd_c
    
    # dump data to conditions
    wing_results = Results(
        compressibility_drag      = cd_c    ,
        thickness_to_chord        = tc      , 
        wing_sweep                = sweep_w , 
        crest_critical            = mcc     ,
        divergence_mach           = MDiv    ,
    )
    drag_breakdown.compressible[wing.tag] = wing_results


    # dump total comp drag
    #drag_breakdown.compressible.total = total_compressibility_drag
    #conditions.aerodynamics.drag_breakdown.compressible[wing.tag] = wing_results
    #conditions.aerodynamics.drag_breakdown.compressible.total = total_compressibility_drag
    
    return total_compressibility_drag
