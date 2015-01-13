""" Constraints.py: Functions defining the constraints on dynamics """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def vertical_force(segment,F_mg=0.0):

    m = segment.m
    m[m < segment.config.Mass_Props.m_empty] = segment.config.Mass_Props.m_empty

    return segment.vectors.Ftot[:,2]/(m*segment.g) - 1.0 - F_mg