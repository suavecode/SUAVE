# compressibility_drag_wing.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Nov 2016, T. MacDonald
#        

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses import Results
from SUAVE.Core import (
    Data, Container
)
from SUAVE.Components import Wings

# package imports
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

def compressibility_drag_wing(state,settings,geometry):
    """ SUAVE.Methods.compressibility_drag_wing(conditions,configuration,geometry)
        computes the induced drag associated with a wing 
        
        Inputs:
        
        Outputs:
        
        Assumptions:
            based on a set of fits
            
    """
    
    # unpack
    conditions    = state.conditions
    configuration = settings    
    
    wing = geometry
    if wing.tag == 'main_wing':
        wing_lifts = conditions.aerodynamics.lift_breakdown.compressible_wings # currently the total aircraft lift
    elif wing.vertical:
        wing_lifts = 0
    else:
        wing_lifts = 0.15 * conditions.aerodynamics.lift_breakdown.compressible_wings
        
    mach           = conditions.freestream.mach_number
    drag_breakdown = conditions.aerodynamics.drag_breakdown
    
    # start result
    total_compressibility_drag = 0.0
        
    # unpack wing
    t_c_w   = wing.thickness_to_chord
    sweep_w = wing.sweeps.quarter_chord
    
    # Currently uses vortex lattice model on all wings
    if wing.tag=='main_wing':
        cl_w = wing_lifts
    else:
        cl_w = 0
        
    cos_sweep = np.cos(sweep_w)

    # get effective Cl and sweep
    tc = t_c_w /(cos_sweep)
    cl = cl_w / (cos_sweep*cos_sweep)

    # compressibility drag based on regressed fits from AA241
    mcc_cos_ws = 0.922321524499352       \
               - 1.153885166170620*tc    \
               - 0.304541067183461*cl    \
               + 0.332881324404729*tc*tc \
               + 0.467317361111105*tc*cl \
               + 0.087490431201549*cl*cl
        
    # crest-critical mach number, corrected for wing sweep
    mcc = mcc_cos_ws / cos_sweep
    
    # divergence mach number
    MDiv = mcc * ( 1.02 + 0.08*(1 - cos_sweep) )
    
    # divergence ratio
    mo_mc = mach/mcc
    
    # compressibility correlation, Shevell
    dcdc_cos3g = 0.0019*mo_mc**14.641
    
    # compressibility drag
    cd_c = dcdc_cos3g * cos_sweep*cos_sweep*cos_sweep
    
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
    
    return total_compressibility_drag
