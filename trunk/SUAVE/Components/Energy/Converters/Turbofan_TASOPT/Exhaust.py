# Exhaust.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data
from SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Pressure_Difference_Set import Pressure_Difference_Set

class Exhaust(Pressure_Difference_Set):
    """Exhaust computations based on TASOPT model"""
    
    def __defaults__(self):
        self.design_polytropic_efficiency = 1.
        self.efficiency_map = None
        self.speed_map      = None
        self.speed_change_by_pressure_ratio = 0.
        self.speed_change_by_mass_flow      = 0.
       
    def compute(self): 
        
        self.compute_flow()
        
        # Assume that pressure difference gives static enthalpy
        Hti = self.inputs.total_enthalpy
        Hf  = self.outputs.total_enthalpy   
        self.outputs.static_enthalpy    = Hf
        self.outputs.static_pressue     = self.outputs.total_pressure
        self.outputs.static_temperature = self.outputs.total_temperature
        
        self.outputs.total_temperature = self.inputs.total_temperature
        self.outputs.total_pressure    = self.inputs.total_pressure
        self.outputs.total_enthalpy    = self.inputs.total_enthalpy

        Tti = self.inputs.total_temperature
        
        # If flow should be choked
        
        gamma = self.inputs.working_fluid.gamma
        R     = self.inputs.working_fluid.R        
        
        flow_speed = np.zeros(np.shape(Hti))
        Tf         = np.zeros(np.shape(Hti))
        R          = np.ones(np.shape(Hti))*R
        
        M = 1.0
        
        Tf[Hf>Hti] = Tti[Hf>Hti]*(1.+(gamma[Hf>Hti]-1.)/2.*M*M)
        flow_speed[Hf>Hti] = np.sqrt(gamma[Hf>Hti]*R[Hf>Hti]*Tf[Hf>Hti])
        
        flow_speed[Hf<=Hti] = np.sqrt(2.*(Hti[Hf<=Hti]-Hf[Hf<=Hti]))
        
        self.outputs.flow_speed = flow_speed