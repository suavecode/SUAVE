
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Attributes import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#   Update Functions
# ----------------------------------------------------------------------

def full_finalize(interface):
    
    configs = interface.configs
    configs.finalize()
    
    analyses = interface.analyses
    analyses.finalize()
    
    strategy = interface.strategy
    strategy.finalize()
    
    
    