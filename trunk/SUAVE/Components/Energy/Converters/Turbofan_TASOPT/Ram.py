# Turbine.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data
from SUAVE.Components.Energy.Converters.Turbofan_TASOPT import Enthalpy_Difference_Set

class Ram(Enthalpy_Difference_Set):
    
    def __defaults__(self):
        pass
    
    def compute(self):
        self.compute_flow()