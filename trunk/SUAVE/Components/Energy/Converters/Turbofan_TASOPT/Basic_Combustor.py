# Basic_Combustor.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component

class Basic_Combustor(Energy_Component):
    """Class used to determine flow properties through a basic combustor."""
    
    def __defaults__(self):
        
        self.tag = 'Basic_Combustor'
        self.inputs.working_fluid = Data()
        self.pressure_ratio = 1.
        self.efficiency = 1.
        
    def compute(self):
         
        Tti = self.inputs.total_temperature
        Pti = self.inputs.total_pressure
        Hti = self.inputs.total_enthalpy
        pi  = self.pressure_ratio
        
        cp = self.inputs.working_fluid.specific_heat
        
        Ptf = Pti*pi
        Ttf = self.turbine_inlet_temperature
        eta = self.efficiency
        hf  = self.fuel_data.specific_energy
        Htf = cp*Ttf
        f = (Htf - Hti)/(eta*hf-Htf)
        
        self.outputs.total_temperature    = Ttf
        self.outputs.total_pressure       = Ptf
        self.outputs.total_enthalpy       = Htf
        self.outputs.normalized_fuel_flow = f