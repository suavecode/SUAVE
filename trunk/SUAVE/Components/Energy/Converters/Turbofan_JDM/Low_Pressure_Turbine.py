# Turbine.py
#
# Created:  Jul 2014, A. Variyar
# Modified: Jan 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

# package imports
import numpy as np
import scipy as sp

from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Turbine Component
# ----------------------------------------------------------------------

class Low_Pressure_Turbine(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Turbine
        a Turbine component
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        #set the default values
        self.tag ='Turbine'
        self.mechanical_efficiency             = 1.0
        self.polytropic_efficiency             = 1.0
        self.inputs.stagnation_temperature     = 1.0
        self.inputs.stagnation_pressure        = 1.0
        self.inputs.fuel_to_air_ratio          = 1.0
        self.outputs.stagnation_temperature    = 1.0
        self.outputs.stagnation_pressure       = 1.0
        self.outputs.stagnation_enthalpy       = 1.0
    
    
    
    
    def compute(self,conditions):
        
        #unpack the values
        
        #unpack from inputs
        tau_r        = self.inputs.tau_r
        tau_cL       = self.inputs.tau_cL
        tau_tH       = self.inputs.tau_tH
        tau_lamda    = self.inputs.tau_lambda
        f            = self.inputs.f
        gamma_t      = self.inputs.working_fluid.gamma
        
        #unpack from self
        eta_m        = self.mechanical_efficiency
        e_t          = self.polytropic_efficiency
        
        #method to compute turbine properties
        
        # Check for bypass
        aalpha = self.inputs.aalpha
        if aalpha == None:
            aalpha = 0
            tau_f = 0
        else:
            tau_f = self.inputs.tau_f
        
        tau_t = 1.0 - tau_r/(eta_m*tau_lamda*tau_tH*(1.0+f))*(tau_cL - 1.0 + aalpha*(tau_f -1.0))
        pi_t = tau_t **(gamma_t/((gamma_t - 1.0)*e_t))
        eta_t = (1.0-tau_t)/(1.0 - tau_t**(1/e_t)) 
        
        #pack the computed values into outputs
        self.outputs.tau_t  = tau_t
        self.outputs.pi_t   = pi_t
        self.outputs.eta_t  = eta_t
    
    
    __call__ = compute