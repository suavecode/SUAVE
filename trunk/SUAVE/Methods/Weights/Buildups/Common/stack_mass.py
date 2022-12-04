## @ingroup Methods-Weights-Buildups-Common
# stack_mass.py
# 
# Created:    Dec 2022, J. Smart
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

import numpy as np

## @ingroup Methods-Weights-Buildups-Common
def stack_mass(materials, *args, **kwargs):
    """Computes the areal mass of a stack of materials

    Assumptions:
    None

    Source:
    None

    Inputs:
    materials                       [SUAVE.Data]
        material.                   [SUAVE.Attributes.Solid]
            minimum_gage_thickness  [m]
            density                 [kg/m^3]

    Outputs: 
    areal_mass  [kg/m^2]

    Properties Used:
    N/A	
    """

    return np.sum([(mat.minimum_gage_thickness * mat.density)
                   for mat in materials])
