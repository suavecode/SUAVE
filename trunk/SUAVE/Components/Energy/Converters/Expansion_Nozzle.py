# Expansion Nozzle.py
#
# Created:  Anil, July 2014


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
from SUAVE.Methods.Propulsion.fm_id import fm_id

# ----------------------------------------------------------------------
#  Expansion Nozzle Component
# ----------------------------------------------------------------------

class Expansion_Nozzle(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Nozzle
        a nozzle component
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        #set the defaults
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
        gamma    = conditions.freestream.isentropic_expansion_factor
        Cp       = conditions.freestream.specific_heat_at_constant_pressure
        Po       = conditions.freestream.pressure
        Pto      = conditions.freestream.stagnation_pressure
        Tto      = conditions.freestream.stagnation_temperature
        R        = conditions.freestream.universal_gas_constant
        Mo       = conditions.freestream.mach_number
        
        #unpack from inputs
        Tt_in    = self.inputs.stagnation_temperature
        Pt_in    = self.inputs.stagnation_pressure
        
        #unpack from self
        pid      = self.pressure_ratio
        etapold  = self.polytropic_efficiency
        
        
        #Method for computing the nozzle properties
        
        #--Getting the output stagnation quantities
        Pt_out   = Pt_in*pid
        Tt_out   = Tt_in*pid**((gamma-1)/(gamma)*etapold)
        ht_out   = Cp*Tt_out
        
        
        #compute the output Mach number, static quantities and the output velocity
        Mach          = np.sqrt((((Pt_out/Po)**((gamma-1)/gamma))-1)*2/(gamma-1))
        T_out         = Tt_out/(1+(gamma-1)/2*Mach**2)
        h_out         = Cp*T_out
        u_out         = np.sqrt(2*(ht_out-h_out))
        
        #Checking from Mach numbers below, above 1.0
        i_low         = Mach < 1.0
        i_high        = Mach >=1.0
        
        #initializing the Pout array
        P_out         = 1.0 *Mach/Mach
        
        #Computing output pressure and Mach number for the case Mach <1.0
        P_out[i_low]  = Po[i_low]
        Mach[i_low]   = np.sqrt((((Pt_out[i_low]/Po[i_low])**((gamma-1)/gamma))-1)*2/(gamma-1))
        
        #Computing output pressure and Mach number for the case Mach >=1.0        
        Mach[i_high]  = 1.0*Mach[i_high]/Mach[i_high]
        P_out[i_high] = Pt_out[i_high]/(1+(gamma-1)/2*Mach[i_high]**2)**(gamma/(gamma-1))
        
        #Computing the output temperature,enthalpy, velocity and density
        T_out         = Tt_out/(1+(gamma-1)/2*Mach**2)
        h_out         = Cp*T_out
        u_out         = np.sqrt(2*(ht_out-h_out))
        rho_out       = P_out/(R*T_out)
        
        #Computing the freestream to nozzle area ratio (mainly from thrust computation)
        area_ratio    = (fm_id(Mo)/fm_id(Mach)*(1/(Pt_out/Pto))*(np.sqrt(Tt_out/Tto)))
        
        #pack computed quantities into outputs
        self.outputs.stagnation_temperature  = Tt_out
        self.outputs.stagnation_pressure     = Pt_out
        self.outputs.stagnation_enthalpy     = ht_out
        self.outputs.mach_number             = Mach
        self.outputs.static_temperature      = T_out
        self.outputs.static_enthalpy         = h_out
        self.outputs.velocity                = u_out
        self.outputs.static_pressure         = P_out
        self.outputs.area_ratio              = area_ratio
    

    __call__ = compute