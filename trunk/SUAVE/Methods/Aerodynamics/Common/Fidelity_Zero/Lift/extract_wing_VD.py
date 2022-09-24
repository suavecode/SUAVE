## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# extract_wing_VD.py
#
# Created:   Feb 2022, R. Erhard
# Modified:  Mar 2022, R. Erhard
#            Apr 2022, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
from SUAVE.Core import Data
import numpy as np
import jax.numpy as jnp

from SUAVE.Analyses.Aerodynamics.Vortex_Lattice import Vortex_Lattice
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift import generate_vortex_distribution


## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def extract_wing_collocation_points(geometry, wing_instance_tag):

    """ This extracts the collocation points of the vehicle vortex distribution
    belonging to the specified wing instance index. This is useful for slipstream
    analysis, where the wake of a propeller is included in the VLM analysis 
    of a specified wing in the vehicle.

    Source:
    None

    Inputs:
    geometry      -    SUAVE vehicle 
    wing_instance -    wing instance tag

    Outputs:
    VD_wing   - colocation points of vortex distribution for specified wing
    ids       - indices in the vortex distribution corresponding to specified wing

    Properties Used:
    N/A
    """
    # unpack vortex distribution properties
    try:
        VD           = geometry.vortex_distribution
    except:
        print("No vortex distribution defined. Creating default vortex distribution from vehicle.")
        settings = Vortex_Lattice().settings
        VD = generate_vortex_distribution(geometry,settings)        
        
    #sym          = VD.symmetric_wings
    
    # Find the beginning and end indices of the wing
    # all breaks
            #breaks = np.hstack([0,np.cumsum(VD.n_cw*VD.n_sw)])
            
            ## Find the initial index of the wing
            #semispan_idx     = wing_instance_idx + np.sum(sym[0:wing_instance_idx])
            #start_pt         = breaks[semispan_idx]
            
            ## Find the final index of the wing
            #end_semispan_idx = semispan_idx + 1 + sym[wing_instance_idx]
            #end_pt           = breaks[end_semispan_idx]
            
            ## Make ranges of points
            #ids              = range(semispan_idx, end_semispan_idx)
            #pt_ids           = range(start_pt,end_pt)
            
    VD_wing = geometry.vortex_distribution.VLM_wings[wing_instance_tag]
    ID      = VD_wing.surface_ID
            
    pt_ids = jnp.where(jnp.abs(VD.surface_ID) == ID,size=VD.surface_ID.shape[0]) 
    

    # Pack the wing level resultss
    VD_wing.XC   = VD.XC[pt_ids]
    VD_wing.YC   = VD.YC[pt_ids]
    VD_wing.ZC   = VD.ZC[pt_ids]
    
    VD_wing.XA1  = VD.XA1[pt_ids]
    VD_wing.XA2  = VD.XA2[pt_ids]
    VD_wing.XB1  = VD.XB1[pt_ids]
    VD_wing.XB2  = VD.XB2[pt_ids]
    VD_wing.YA1  = VD.YA1[pt_ids]
    VD_wing.YA2  = VD.YA2[pt_ids]
    VD_wing.YB1  = VD.YB1[pt_ids]
    VD_wing.YB2  = VD.YB2[pt_ids]
    VD_wing.ZA1  = VD.ZA1[pt_ids]
    VD_wing.ZA2  = VD.ZA2[pt_ids]
    VD_wing.ZB1  = VD.ZB1[pt_ids]
    VD_wing.ZB2  = VD.ZB2[pt_ids]    
    
    
    VD_wing.n_cp = len(VD_wing.XC)
    #VD_wing.n_cw = VD.n_cw[ids]
    #VD_wing.n_sw = VD.n_sw[ids]
    #VD_wing.static_keys = ['n_cw','n_sw','wing_type']

    return VD_wing, pt_ids


