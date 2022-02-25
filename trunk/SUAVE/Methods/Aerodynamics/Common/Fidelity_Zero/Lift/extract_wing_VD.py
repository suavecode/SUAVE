## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# extract_wing_collocation_points.py
#
# Created:   Feb 2022, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from SUAVE.Core import Data

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def extract_wing_collocation_points(geometry, wing_instance_idx):

    """ This extracts the collocation points of the vehicle vortex distribution
    belonging to the specified wing instance index. This is used for 

    Source:
    None

    Inputs:
    geometry      -    SUAVE vehicle 
    wing_instance -    wing instance to extract VD for

    Outputs:
    None

    Properties Used:
    N/A
    """
    # unpack vortex distribution properties
    VD   = geometry.vortex_distribution
    n_sw = VD.n_sw 
    n_cw = VD.n_cw  

    VD_wing = Data()    
    vd_i    = 0        # count of current VD wing elements
    j       = 0        # count of current VD wing index
    size = n_cw * n_sw
    
    for idx,wing in enumerate(geometry.wings):
        
        if wing.symmetric:
            wing_cp_size = size[j] + size[j+1]
            j += 2
        else:
            wing_cp_size = size[j]
            j += 1
            
        if idx == wing_instance_idx:
            # store the VD corresponding to this wing
            VD_wing.XC = VD.XC[vd_i : vd_i + wing_cp_size]
            VD_wing.YC = VD.YC[vd_i : vd_i + wing_cp_size]
            VD_wing.ZC = VD.ZC[vd_i : vd_i + wing_cp_size]
            
            ids = (np.linspace(vd_i, vd_i+wing_cp_size-1,  wing_cp_size)).astype(int)
                   
        vd_i += wing_cp_size
        
    # extract VD elements for vd_ele

    

    return VD_wing, ids


