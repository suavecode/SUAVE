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

    return np.sum([(mat.minimum_gage_thickness * mat.density) for mat in materials])
