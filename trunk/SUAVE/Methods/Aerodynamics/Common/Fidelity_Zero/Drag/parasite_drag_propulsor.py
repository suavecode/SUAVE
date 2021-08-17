## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
# parasite_drag_propulsor.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Jan 2016, E. Botero 
#         

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Core import Data
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions import compressible_turbulent_flat_plate

# package imports
import numpy as np

# ----------------------------------------------------------------------
#   Parasite Drag Propulsor
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Drag
def parasite_drag_propulsor(state,settings,geometry):
    """Computes the parasite drag due to the network

    Assumptions:
    Basic fit

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)
    Raymer: Aircraft Design: A Conceptual Approach

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
    configuration = settings
    
    network   = geometry
    Sref      = network.nacelle_diameter**2. / 4. * np.pi
    Swet      = network.areas.wetted
    
    l_prop = network.engine_length
    d_prop = network.nacelle_diameter
    
    # conditions
    freestream = conditions.freestream
    Mc  = freestream.mach_number
    Tc  = freestream.temperature    
    re  = freestream.reynolds_number

    # reynolds number
    Re_prop = re*l_prop
    
    # skin friction coefficient
    cf_prop, k_comp, k_reyn = compressible_turbulent_flat_plate(Re_prop,Mc,Tc)
    
    ## form factor according to Raymer equation (pg 283 of Aircraft Design: A Conceptual Approach)
    k_prop = 1 + 0.35 / (l_prop/d_prop)
    
   
    # find the final result    
    network_parasite_drag = k_prop * cf_prop * Swet / Sref
    
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
    conditions.aerodynamics.drag_breakdown.parasite[network.tag] = network_result    
    
    return network_parasite_drag