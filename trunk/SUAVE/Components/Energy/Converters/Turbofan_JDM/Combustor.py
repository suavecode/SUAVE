# Combustor.py
#
# Created:  Oct 2014, A. Variyar
# Modified: Jan 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component


# ----------------------------------------------------------------------
#  Combustor Component
# ----------------------------------------------------------------------

class Combustor(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Combustor
        a combustor component
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        
        self.tag = 'Combustor'
        
        #-----setting the default values for the different components
        self.fuel_data                      = SUAVE.Attributes.Propellants.Jet_A()
        self.alphac                         = 0.0
        self.turbine_inlet_temperature      = 1.0
        self.inputs.stagnation_temperature  = 1.0
        self.inputs.stagnation_pressure     = 1.0
        self.outputs.stagnation_temperature = 1.0
        self.outputs.stagnation_pressure    = 1.0
        self.outputs.stagnation_enthalpy    = 1.0
        self.outputs.fuel_to_air_ratio      = 1.0
        self.fuel_data                      = Data()
    
    
    
    def compute(self,conditions):
        
        # unpack the values
        tau_lamda = self.inputs.tau_lambda
        tau_f     = self.inputs.tau_f     
        tau_r     = self.inputs.tau_r     
        tau_cL    = self.inputs.tau_cL    
        tau_cH    = self.inputs.tau_cH    
        c_pc      = self.inputs.c_pc        
        
        # unpacking the values from conditions
        T0     = conditions.freestream.temperature
        
        # unpacking values from self
        h_pr    = self.fuel_data.specific_energy  
        eta_b   = self.efficiency

        # Calculate fuel flow
        f = (tau_lamda - tau_f*tau_r*tau_cH*tau_cL)/((eta_b*h_pr/(c_pc*T0)) - tau_lamda)
        
        # pack computed quantities into outputs
        self.outputs.fuel_to_air_ratio       = f 
    
    
    
    __call__ = compute
