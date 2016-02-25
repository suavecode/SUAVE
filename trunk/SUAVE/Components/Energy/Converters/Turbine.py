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

class Turbine(Energy_Component):
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
        
        #unpack from conditions
        gamma           = conditions.freestream.isentropic_expansion_factor
        Cp              = conditions.freestream.specific_heat_at_constant_pressure
        
        #unpack from inputs
        Tt_in           = self.inputs.stagnation_temperature
        Pt_in           = self.inputs.stagnation_pressure
        alpha           =  self.inputs.bypass_ratio
        f               = self.inputs.fuel_to_air_ratio
        compressor_work = self.inputs.compressor.work_done
        fan_work        = self.inputs.fan.work_done
        
        #unpack from self
        eta_mech        =  self.mechanical_efficiency
        etapolt         =  self.polytropic_efficiency
        
        #method to compute turbine properties
        
        #Using the work done by the compressors/fan and the fuel to air ratio to compute the energy drop across the turbine
        deltah_ht =  -1/(1+f)*1/eta_mech*((compressor_work)+ alpha*(fan_work))
        
        #Compute the output stagnation quantities from the inputs and the energy drop computed above
        Tt_out    =  Tt_in+deltah_ht/Cp
        Pt_out    =  Pt_in*(Tt_out/Tt_in)**(gamma/((gamma-1)*etapolt))
        ht_out    =  Cp*Tt_out   #h(Tt4_5)
        
        
        #pack the computed values into outputs
        self.outputs.stagnation_temperature  = Tt_out
        self.outputs.stagnation_pressure     = Pt_out
        self.outputs.stagnation_enthalpy     = ht_out
    
    
    __call__ = compute