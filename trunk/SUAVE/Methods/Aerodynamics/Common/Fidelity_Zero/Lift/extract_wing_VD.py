## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# extract_wing_collocation_points.py
#
# Created:   Feb 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
from SUAVE.Core import Data

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
    VD          = geometry.vortex_distribution
    span_breaks = VD.spanwise_breaks
    wing        = list(geometry.wings.keys())[wing_instance_idx]
    sym         = geometry.wings[wing].symmetric
    
    VD_wing  = Data()       
    id_start = span_breaks[wing_instance_idx]
    id_end   = span_breaks[wing_instance_idx + 1 + int(sym)]
    ids      = range(id_start,id_end)
    
    VD_wing.XC   = VD.XC[ids]
    VD_wing.YC   = VD.YC[ids]
    VD_wing.ZC   = VD.ZC[ids]
    VD_wing.n_cp = len(VD_wing.XC)

    return VD_wing, ids


