## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# extract_wing_VD.py
#
# Created:   Feb 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
from SUAVE.Core import Data
import numpy as np

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def extract_wing_collocation_points(geometry, wing_instance_idx):

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
    VD           = geometry.vortex_distribution
    span_breaks  = VD.spanwise_breaks
    sym          = VD.symmetric_wings
    
    VD_wing  = Data()       
    semispan_idx = wing_instance_idx + np.sum(sym[0:wing_instance_idx])
    id_start = span_breaks[semispan_idx]
    id_end   = span_breaks[semispan_idx + 1 + sym[wing_instance_idx]]
    
    ids      = range(semispan_idx, semispan_idx + 1 + sym[wing_instance_idx])
    n_cw     = VD.n_cw[ids]

    if bool(sym[wing_instance_idx]):
        pt_ids   = range(id_start*n_cw[0],id_end*n_cw[1])
    else:
        pt_ids   = range(id_start*n_cw[0],id_end*n_cw[0])
        
    
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
    VD_wing.n_cw = VD.n_cw[ids]
    VD_wing.n_sw = VD.n_sw[ids]

    return VD_wing, pt_ids


