# Trust_Region.py
#
# Created:  Apr 2017, T. MacDonald
# Modified: Jun 2017, T. MacDonald

import SUAVE
from SUAVE.Core import Data
import numpy as np
import copy

class Trust_Region(Data):
    
    def __defaults__(self):
        
        self.initial_size       = 0.05
        self.size               = 0.05
        self.minimum_size       = 1e-15
        self.contract_threshold = 0.25
        self.expand_threshold   = 0.75
        self.contraction_factor = 0.25
        self.expansion_factor   = 1.5
        
        
    def evaluate_function(self,f,gviol):
        phi = f + gviol**2
        return phi        