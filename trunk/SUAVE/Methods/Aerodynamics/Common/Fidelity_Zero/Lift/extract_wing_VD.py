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

    # unpack
    VD  = geometry.vortex_distribution
    n_w  = VD.n_w
    n_cp = VD.n_cp
    n_sw = VD.n_sw 
    n_cw = VD.n_cw  
    
    VD_wing = Data()
    VD_wing.XC = VD.XC
    VD_wing.YC = VD.YC
    VD_wing.ZC = VD.ZC
    

    return VD_wing


