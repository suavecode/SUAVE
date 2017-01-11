# parasite_drag_propulsor.py
# 
# Created:  Aug 2014, T. MacDonald
# Modified: Nov 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from compressible_turbulent_flat_plate import compressible_turbulent_flat_plate
from SUAVE.Analyses import Results

import numpy as np

# ----------------------------------------------------------------------
#   Parasite Drag Propulsors
# ----------------------------------------------------------------------

def parasite_drag_propulsor(state,settings,geometry):
    """ SUAVE.Methods.parasite_drag_propulsor(conditions,configuration,propulsor)
        computes the parasite drag associated with a propulsor 
        
        Inputs:

        Outputs:

        Assumptions:

        
    """
    
    # unpack inputs
    
    conditions    = state.conditions
    configuration = settings
    propulsor     = geometry
        
    freestream = conditions.freestream
    
    Sref        = propulsor.nacelle_diameter**2 / 4 * np.pi
    Swet        = propulsor.areas.wetted
    
    l_prop  = propulsor.engine_length
    d_prop  = propulsor.nacelle_diameter
    
    # conditions
    freestream = conditions.freestream
    Mc = freestream.mach_number
    Tc = freestream.temperature    
    re = freestream.reynolds_number

    # reynolds number
    Re_prop = re*l_prop
    
    # skin friction coefficient
    cf_prop, k_comp, k_reyn = compressible_turbulent_flat_plate(Re_prop,Mc,Tc)


    k_prop = np.array([[0.0]]*len(Mc))
    # assume that the drag divergence mach number of the propulsor matches the main wing
    Mdiv = state.conditions.aerodynamics.drag_breakdown.compressible.main_wing.divergence_mach
    
    # form factor according to Raymer equation (pg 283 of Aircraft Design: A Conceptual Approach)
    k_prop_sub = 1. + 0.35 / (float(l_prop)/float(d_prop)) 
    
    # for supersonic flow (http://adg.stanford.edu/aa241/drag/BODYFORMFACTOR.HTML)
    k_prop_sup = 1.
    
    sb_mask = (Mc <= Mdiv)
    tn_mask = ((Mc > Mdiv) & (Mc < 1.05))
    sp_mask = (Mc >= 1.05)
    
    k_prop[sb_mask] = k_prop_sub
    # basic interpolation for transonic
    k_prop[tn_mask] = (k_prop_sup-k_prop_sub)*(Mc[tn_mask]-Mdiv[tn_mask])/(1.05-Mdiv[tn_mask]) + k_prop_sub
    k_prop[sp_mask] = k_prop_sup
    
    # --------------------------------------------------------
    # find the final result    
    propulsor_parasite_drag = k_prop * cf_prop * Swet / Sref  
    # --------------------------------------------------------
    
    # dump data to conditions
    propulsor_result = Results(
        wetted_area               = Swet    , 
        reference_area            = Sref    , 
        parasite_drag_coefficient = propulsor_parasite_drag ,
        skin_friction_coefficient = cf_prop ,
        compressibility_factor    = k_comp  ,
        reynolds_factor           = k_reyn  , 
        form_factor               = k_prop  ,
    )
    state.conditions.aerodynamics.drag_breakdown.parasite[propulsor.tag] = propulsor_result    
    
    return propulsor_parasite_drag