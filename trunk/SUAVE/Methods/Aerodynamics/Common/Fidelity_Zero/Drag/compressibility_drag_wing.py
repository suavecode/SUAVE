## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
# compressibility_drag_wing.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Nov 2016, T. MacDonald
#           Apr 2020, M. Clarke        
#           Apr 2020, M. Clarke
#           May 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Core import Data
from SUAVE.Components import Wings

# package imports
import numpy as np
import scipy as sp


# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
def compressibility_drag_wing(state,settings,geometry):
    """Computes compressibility drag for a wing

    Assumptions:
    Subsonic to low transonic
    Supercritical airfoil

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)

    Inputs:
    state.conditions.
      freestream.mach_number                         [Unitless]
      aerodynamics.lift_breakdown.compressible_wings [Unitless]
    geometry.thickness_to_chord                      [Unitless]
    geometry.sweeps.quarter_chord                    [radians]

    Outputs:
    total_compressibility_drag                       [Unitless]

    Properties Used:
    N/A
    """ 
    
    # unpack
    conditions     = state.conditions
    wing           = geometry
    cl_w           = conditions.aerodynamics.lift_breakdown.compressible_wings[wing.tag]         
    mach           = conditions.freestream.mach_number
    drag_breakdown = conditions.aerodynamics.drag_breakdown

    # unpack wing
    t_c_w   = wing.thickness_to_chord
    sweep_w = wing.sweeps.quarter_chord
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

    # dump data to conditions
    wing_results = Data(
        compressibility_drag      = cd_c    ,
        thickness_to_chord        = tc      , 
        wing_sweep                = sweep_w , 
        crest_critical            = mcc     ,
        divergence_mach           = MDiv    ,
    )
    drag_breakdown.compressible[wing.tag] = wing_results
