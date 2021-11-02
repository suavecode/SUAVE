## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
# parasite_drag_propulsor.py
# 
# Created:  Feb 2019, T. MacDonald
# Modified: Jan 2020, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions import compressible_turbulent_flat_plate
from SUAVE.Core import Data
from SUAVE.Methods.Utilities.Cubic_Spline_Blender import Cubic_Spline_Blender

import numpy as np

# ----------------------------------------------------------------------
#   Parasite Drag Propulsors
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Supersonic_Zero-Drag
def parasite_drag_propulsor(state,settings,geometry):
    """Computes the parasite drag due to the network

    Assumptions:
    Basic fit

    Source:
    Raymer equation (pg 283 of Aircraft Design: A Conceptual Approach) (subsonic)
    http://aerodesign.stanford.edu/aircraftdesign/drag/BODYFORMFACTOR.HTML (supersonic)

    Inputs:
    state.conditions.freestream.
      mach_number                                [Unitless]
      temperature                                [K]
      reynolds_number                            [Unitless]
    geometry.      
      nacelle_diameter                           [m^2]
      areas.wetted                               [m^2]
      engine_length                              [m]

    Outputs:
    network_parasite_drag                      [Unitless]

    Properties Used:
    N/A
    """
    
    # unpack inputs
    
    conditions    = state.conditions
    network       = geometry
    
    low_mach_cutoff  = settings.begin_drag_rise_mach_number
    high_mach_cutoff = settings.end_drag_rise_mach_number    
        
    freestream = conditions.freestream
    
    Sref        = network.nacelle_diameter**2 / 4 * np.pi
    Swet        = network.areas.wetted
    
    l_prop  = network.engine_length
    d_prop  = network.nacelle_diameter
    
    # conditions
    freestream = conditions.freestream
    Mc = freestream.mach_number
    Tc = freestream.temperature    
    re = freestream.reynolds_number

    # reynolds number
    Re_prop = re*l_prop
    
    # skin friction coefficient
    cf_prop, k_comp, k_reyn = compressible_turbulent_flat_plate(Re_prop,Mc,Tc)

    
    # form factor according to Raymer equation (pg 283 of Aircraft Design: A Conceptual Approach)
    k_prop_sub = 1. + 0.35 / (float(l_prop)/float(d_prop)) 
    
    # for supersonic flow (http://adg.stanford.edu/aa241/drag/BODYFORMFACTOR.HTML)
    k_prop_sup = 1.
    
    trans_spline = Cubic_Spline_Blender(low_mach_cutoff,high_mach_cutoff)
    h00 = lambda M:trans_spline.compute(M)
    
    k_prop = k_prop_sub*(h00(Mc)) + k_prop_sup*(1-h00(Mc))
    
    # --------------------------------------------------------
    # find the final result    
    network_parasite_drag = k_prop * cf_prop * Swet / Sref  
    # --------------------------------------------------------
    
    # dump data to conditions
    network_result = Data(
        wetted_area               = Swet    , 
        reference_area            = Sref    , 
        parasite_drag_coefficient = network_parasite_drag ,
        skin_friction_coefficient = cf_prop ,
        compressibility_factor    = k_comp  ,
        reynolds_factor           = k_reyn  , 
        form_factor               = k_prop  ,
    )
    state.conditions.aerodynamics.drag_breakdown.parasite[network.tag] = network_result    
    
    return network_parasite_drag