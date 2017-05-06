# compressibility_drag_wing.py
# 
# Created:  Your Name, Dec 2013
# Modified:         

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Analyses import Results
from SUAVE.Core import (
    Data, Container,  
)
from SUAVE.Components import Wings

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn
import copy

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
    
    wing       = geometry
    if isinstance(wing,Wings.Main_Wing):
        wing_lifts = conditions.aerodynamics.lift_breakdown.compressible_wings # currently the total aircraft lift
    elif wing.vertical:
        wing_lifts = 0
    else:
        wing_lifts = 0.0 * conditions.aerodynamics.lift_breakdown.compressible_wings
        
    mach       = conditions.freestream.mach_number
    drag_breakdown = conditions.aerodynamics.drag_breakdown
    

    # start result
    total_compressibility_drag = 0.0
    #drag_breakdown.compressible = Results()

    
        
    # unpack wing
    t_c_w   = wing.thickness_to_chord
    sweep_w = wing.sweeps.quarter_chord
    
    # Currently uses vortex lattice model on all wings
    if wing.tag=='main_wing':
        cl_w = wing_lifts
    else:
        cl_w = 0

    # get effective Cl and sweep
    tc = t_c_w /(np.cos(sweep_w))
    cl = cl_w / (np.cos(sweep_w))**2
    




    #-----------Existing SUAVE-----------------------------------


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
    
    #-----------------------------------------------------------------







    ##----------------------------------------------------------------------------------






    
    
    ##modified implementation -------------------------------------------------------
    
    supercrit = 1.0    
    
    mcc = (0.95399999999999996 - 0.23499999999999999*cl)+0.025899999999999999*cl*cl
    mcc = mcc - ((1.9630000000000001 - 1.0780000000000001 * cl) + 0.34999999999999998 * cl * cl) * tc
    mcc = mcc + ((2.9689999999999999 - 2.738 * cl) + 1.4690000000000001 * cl * cl) * tc * tc
    mcc = mcc + supercrit * 0.059999999999999998
    mcc = mcc / np.cos(sweep_w)
    rm = mach / mcc
    dm = rm - 1.0
    
    cd_c = copy.deepcopy(mach)
        

    for irm in range(0,len(rm)): 
        if((rm[irm] >= 0.5) and (rm[irm] < 0.80000000000000004)):
            cd_c[irm] = 0.00013888951999999999 + 0.00055555999999999997 * dm[irm] + 0.00055556192 * dm[irm] * dm[irm]
        elif((rm[irm] >= 0.80000000000000004) and (rm[irm] < 0.94999999999999996)):
            cd_c[irm] = 0.00070899999999999999 + 0.0067330000000000003 * dm[irm] + 0.019560000000000001 * dm[irm] * dm[irm] + 0.011849999999999999 * dm[irm] * dm[irm] * dm[irm]

        elif((rm[irm] >= 0.94999999999999996) and (rm[irm] < 1.0)):
            cd_c[irm] = 0.001 + 0.027269999999999999 * dm[irm] + 0.49199999999999999 * dm[irm] * dm[irm] + 3.5738500000000002 * dm[irm] * dm[irm] * dm[irm]
        elif(rm[irm] >= 1.0):
            cd_c[irm] = ((0.001 + 0.027269999999999999 * dm[irm]) - 0.19520000000000001 * dm[irm] * dm[irm]) + 19.09 * dm[irm] * dm[irm] * dm[irm]   
            
        else:
            cd_c[irm] = 0.0
    
    
    
    cd_c = cd_c * np.cos(sweep_w)**3.0 
    mdiv = mcc * (1.02 + (1.0 - np.cos(sweep_w)) * 0.080000000000000002)
    
    #cd_c = cd_c.tolist()

    MDiv  = mdiv    

    
    ##----------------------------------------------------------------------------------

    
    
    
    
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
