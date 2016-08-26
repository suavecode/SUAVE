# Cooling_Combustor.py
#
# Created:  Jul 2016, T. MacDonald
# Modified:

# SUAVE imports

import SUAVE

# package imports
import numpy as np

from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component

class Cooling_Combustor(Energy_Component):
    """Class used to determine flow properties through a basic combustor."""
    
    def __defaults__(self):
        
        self.tag = 'Combustor'
        self.inputs.working_fluid = Data()
        self.film_effectiveness_factor = 0.4 # theta_f in TASOPT manual
        self.weighted_stanton_number = 0.035 # St_A 
        self.cooling_efficiency = 0.7 # eta
        self.delta_temperature_streak = 200. # delta T_streak
        self.metal_temperature = 1400. # T_m
        self.mixing_zone_start_mach_number = 0.8 # M4a
        self.cooling_flow_velocity_ratio = 0.9 # r_u_c
        self.blade_row_exit_mach_number = 0.8 # M_exit
        self.turbine_inlet_temperature  = 1400.
        self.efficiency = 1.
        self.pressure_ratio = 1.
        
    def compute(self):
         
        Tti = self.inputs.total_temperature
        Pti = self.inputs.total_pressure
        Hti = self.inputs.total_enthalpy
        pi  = self.pressure_ratio
        
        cp    = self.inputs.working_fluid.specific_heat
        gamma = self.inputs.working_fluid.gamma
        R     = self.inputs.working_fluid.R
        
        theta_f = self.film_effectiveness_factor
        St_A    = self.weighted_stanton_number
        eta_cf  = self.cooling_efficiency
        eta_b   = self.efficiency
        dTemp_steak = self.delta_temperature_streak
        T_m     = self.metal_temperature
        M4a     = self.mixing_zone_start_mach_number
        Mexit   = self.blade_row_exit_mach_number
        ruc     = self.cooling_flow_velocity_ratio
        
        Tt4     = self.turbine_inlet_temperature
        
        # Simplifying assumption of only one cooling stage
        Tg_1 = Tt4 + dTemp_steak
        theta_1 = (Tg_1-T_m)/(Tg_1-Tti) # Tti should be Tt3
        cooling_mass_flow_ratio = 1./(1.+1./St_A*(eta_cf*(1.-theta_1))/(theta_1*(1.-eta_cf*theta_f)-theta_f*(1.-eta_cf)))
        
        hf  = self.fuel_data.specific_energy
        Ht4 = cp*Tt4
        f   = (Ht4-Hti)*(1-cooling_mass_flow_ratio)/(eta_b*hf - Ht4)
        Pt4 = Pti*pi
        
        Tt4_1 = (hf*f/cp + Tti)/(1.+f)
        u4a   = M4a/np.sqrt(1.+(gamma-1.)/2.*M4a*M4a)*np.sqrt(gamma*R*Tt4)
        uc    = ruc*u4a
    
        u4_1 = ((1.-cooling_mass_flow_ratio+f)*u4a + cooling_mass_flow_ratio*uc)/(1.+f)
        # note that a u4a term appearing in the TASOPT manual has been removed and assumed
        # to be an error. The units do not work out in the TASOPT equation
        T4_1 = Tt4_1 - .5*u4_1*u4_1/cp
        P4_1 = Pt4*((1.+(gamma-1.)/2.*M4a*M4a)**(-gamma/(gamma-1)))
        
        Pt4_1 = P4_1*((Tt4_1/T4_1)**(gamma/(gamma-1)))
        Ht4_1 = cp*Tt4_1
        
        Ttf = Tt4_1
        Ptf = Pt4_1
        Htf = Ht4_1
        
        self.outputs.total_temperature    = Ttf
        self.outputs.total_pressure       = Ptf
        self.outputs.total_enthalpy       = Htf
        self.outputs.normalized_fuel_flow = f