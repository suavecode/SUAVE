# Compression_Nozzle.py
#
# Created:  Anil, July 2014

#--put in a folder

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Attributes import Units

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp


from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Components import Component_Exception
from SUAVE.Components.Propulsors.Propulsor import Propulsor

# ----------------------------------------------------------------------
#  Compression Nozzle Component
# ----------------------------------------------------------------------

class Compression_Nozzle(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Nozzle
        a nozzle component
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        #setting the default values 
        self.tag = 'Nozzle'
        self.polytropic_efficiency           = 1.0
        self.pressure_ratio                  = 1.0
        self.inputs.stagnation_temperature   = 0.
        self.inputs.stagnation_pressure      = 0.
        self.outputs.stagnation_temperature  = 0.
        self.outputs.stagnation_pressure     = 0.
        self.outputs.stagnation_enthalpy     = 0.
    

    def compute(self,conditions):
        
        #unpack the values
        
        #unpack from conditions
        gamma   = conditions.freestream.isentropic_expansion_factor
        Cp      = conditions.freestream.specific_heat_at_constant_pressure
        Po      = conditions.freestream.pressure
        R       = conditions.freestream.universal_gas_constant
        
        #unpack from inpust
        Tt_in   = self.inputs.stagnation_temperature
        Pt_in   = self.inputs.stagnation_pressure
        
        #unpack from self
        pid     =  self.pressure_ratio
        etapold =  self.polytropic_efficiency
        
        #Method to compute the output variables
        
        #--Getting the output stagnation quantities
        Pt_out  = Pt_in*pid
        Tt_out  = Tt_in*pid**((gamma-1)/(gamma*etapold))
        ht_out  = Cp*Tt_out
        
        #compute the output Mach number, static quantities and the output velocity
        Mach    = np.sqrt((((Pt_out/Po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        T_out   = Tt_out/(1+(gamma-1)/2*Mach**2)
        h_out   = Cp*T_out
        u_out   = np.sqrt(2*(ht_out-h_out))
          
        #pack computed quantities into outputs
        self.outputs.stagnation_temperature  = Tt_out
        self.outputs.stagnation_pressure     = Pt_out
        self.outputs.stagnation_enthalpy     = ht_out
        self.outputs.mach_number             = Mach
        self.outputs.static_temperature      = T_out
        self.outputs.static_enthalpy         = h_out
        self.outputs.velocity                = u_out
    

    __call__ = compute