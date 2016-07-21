# Compressor.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data
from SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Pressure_Difference_Set import Pressure_Difference_Set

class Compressor(Pressure_Difference_Set):
    """Turbine computations based on TASOPT model"""
    
    def __defaults__(self):
        self.design_polytropic_efficiency = 1.
        self.design_pressure_ratio        = 1.5
        self.efficiency_map = None
        self.speed_map      = None
        self.speed_change_by_pressure_ratio = 0.
        self.speed_change_by_mass_flow      = 0.
       
    def compute(self):
        
        self.compute_flow()
        
    def compute_performance(self):
        
        # This will change the efficiency
        pi = self.pressure_ratio
        mdotc = self.corrected_mass_flow
        self.polytropic_efficiency = self.efficiency_map.compute_efficiency(pi,mdotc)
        N, dN_pi, dN_mf           = self.speed_map.compute_speed(pi,mdotc)
        self.corrected_speed      = N
        self.speed_change_by_pressure_ratio = dN_pi
        self.speed_change_by_mass_flow      = dN_mf

    def set_design_condition(self):
        
        self.polytropic_efficiency = self.design_polytropic_efficiency
        self.pressure_ratio        = self.design_pressure_ratio
        
    def size(self,mdot,mach_number,bypass_ratio = 0.,hub_to_tip_ratio = 0.):
        
        gamma = self.inputs.working_fluid.gamma
        R     = self.inputs.working_fluid.R
        Tt    = self.inputs.total_temperature
        Pt    = self.inputs.total_pressure
        ht    = self.inputs.total_enthalpy
        
        T = Tt/(1.+((gamma-1.)/2. *mach_number**2.))
        P = Pt/((1.+(gamma-1.)/2. *mach_number**2. )**(gamma/(gamma-1.)))
        h = ht*T/Tt
        
        rho = P/(R*T)
        u   = mach_number*np.sqrt(gamma*R*T)
        A   = (1.+bypass_ratio)*mdot/(rho*u)
        d   = np.sqrt(4./np.pi*A/(1.-hub_to_tip_ratio*hub_to_tip_ratio))
        
        self.entrance_area     = A
        self.entrance_diameter = d