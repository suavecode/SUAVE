""" Propulsion.py: Methods for Propulsion Analysis """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Structure import Data
from SUAVE.Attributes import Constants
# import SUAVE

# ----------------------------------------------------------------------
#  Mission Methods
# ----------------------------------------------------------------------

def fm_id(M):

    R=287.87
    g=1.4
    m0=(g+1)/(2*(g-1))
    m1=((g+1)/2)**m0
    m2=(1+(g-1)/2*M**2)**m0
    fm=m1*M/m2
    return fm