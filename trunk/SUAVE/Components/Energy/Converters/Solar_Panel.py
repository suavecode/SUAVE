#Solar_Panel.py
# 
# Created:  Emilio Botero, Jun 2014
# Modified:  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
from SUAVE.Core import Units
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Solar_Panel Class
# ----------------------------------------------------------------------
class Solar_Panel(Energy_Component):
    
    def __defaults__(self):
        self.area       = 0.0
        self.efficiency = 0.0
    
    def power(self):
        
        # Unpack
        flux       = self.inputs.flux
        efficiency = self.efficiency
        area       = self.area
        
        p = flux*area*efficiency
        
        # Store to outputs
        self.outputs.power = p
    
        return p
    
    
    
    