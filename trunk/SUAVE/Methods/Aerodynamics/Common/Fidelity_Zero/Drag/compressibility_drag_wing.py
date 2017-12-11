## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
# compressibility_drag_wing.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Nov 2016, T. MacDonald
#        

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
    conditions    = state.conditions
    configuration = settings    # unused
    
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
    
    if len(wing.Segments.keys())>0: # if wing has segments
        symm           = wing.symmetric
        semispan       = wing.spans.projected*0.5 * (2 - symm)
        root_chord     = wing.chords.root
        num_segments   = len(wing.Segments.keys())     
        
        weighted_sweep = 0
        Sref           = 0
        for i_segs in xrange(num_segments): 
            if i_segs == num_segments-1:
                continue 
            else:                    
                span_seg        = semispan*(wing.Segments[i_segs+1].percent_span_location - wing.Segments[i_segs].percent_span_location )
                chord_root      = root_chord*wing.Segments[i_segs].root_chord_percent
                chord_tip       = root_chord*wing.Segments[i_segs+1].root_chord_percent
                Sref_seg        = span_seg *(chord_root+chord_tip)*0.5
                weighted_sweep += wing.Segments[i_segs].sweeps.quarter_chord*Sref_seg 
                Sref           += Sref_seg       
        sweep_w = weighted_sweep/Sref
        
  
    else: # if wing does not have segments          
        sweep_w = wing.sweeps.quarter_chord
    
    # compute compressibility drag  
    cd_c, mcc, MDiv ,tc = compute_compressibility_drag (sweep_w,t_c_w,wing,wing_lifts,mach)
    
    # dump data to conditions
    wing_results = Data(
        compressibility_drag      = cd_c    ,
        thickness_to_chord        = tc      , 
        wing_sweep                = sweep_w , 
        crest_critical            = mcc     ,
        divergence_mach           = MDiv    ,
    )
    drag_breakdown.compressible[wing.tag] = wing_results
    
    return total_compressibility_drag

def  compute_compressibility_drag (sweep_w, t_c_w,wing,wing_lifts,mach):
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

    return cd_c, mcc, MDiv ,tc