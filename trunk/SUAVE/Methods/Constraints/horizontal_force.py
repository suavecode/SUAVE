""" Constraints.py: Functions defining the constraints on dynamics """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def horizontal_force(segment,F_mg=0.0):

    m = segment.m
    m[m < segment.config.Mass_Props.m_empty] = segment.config.Mass_Props.m_empty

    return segment.vectors.Ftot[:,0]/(m*segment.g) - F_mg
