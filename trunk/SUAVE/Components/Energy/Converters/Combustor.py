# Gas_Turbine.py
#
# Created:  Anil, July 2014

#--put in a folder

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Structure import Data
from SUAVE.Attributes import Units

# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp

from SUAVE.Structure import (
                             Data, Container, Data_Exception, Data_Warning,
                             )

from SUAVE.Structure import Data, Data_Exception, Data_Warning
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Components import Component_Exception
#from SUAVE.Components.Energy.Gas_Turbine import Network
from SUAVE.Components.Propulsors.Propulsor import Propulsor

class Combustor(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Combustor
        a combustor component
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        
        self.tag = 'Combustor'
        
        self.alphac = 0.0
        self.turbine_inlet_temperature = 1.0
        self.fuel_data = SUAVE.Attributes.Propellants.Jet_A()
        
        self.inputs.stagnation_temperature = 1.0
        self.inputs.stagnation_pressure = 1.0
        #self.inputs.nozzle_temp = 1.0
        
        self.outputs.stagnation_temperature=1.0
        self.outputs.stagnation_pressure=1.0
        self.outputs.stagnation_enthalpy=1.0
        self.outputs.fuel_to_air_ratio = 1.0
    
    
    
    
    
    def compute(self,conditions):
        
        #unpack the variables
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp
        To = conditions.freestream.temperature
        Tto = conditions.freestream.stagnation_temperature
        
        Tt_in = self.inputs.stagnation_temperature
        Pt_in = self.inputs.stagnation_pressure
        Tt_n = self.inputs.nozzle_exit_stagnation_temperature
        Tt4 = self.turbine_inlet_temperature
        pib = self.pressure_ratio
        eta_b = self.efficiency
        #Tto = self.inputs.freestream_stag_temp
        
        htf=self.fuel_data.specific_energy
        ht4 = Cp*Tt4
        ho = Cp*To
        
        
        
        
        
        #Using the Turbine exit temperature, the fuel properties and freestream temperature to compute the fuel to air ratio f
        tau = htf/(Cp*To)
        tau_freestream=Tto/To
        
        
        #f=(((self.Tt4/To)-tau_freestream*(Tt_in/Tt_n))/(self.eta*tau-(self.Tt4/To)))
        
        f = (ht4 - ho)/(eta_b*htf-ht4)
        #print ((self.Tt4/To)-tau_freestream*(Tt_in/Tt_n))
        
        ht_out=Cp*Tt4
        Pt_out=Pt_in*pib
        
        #pack outputs
        self.outputs.stagnation_temperature=Tt4
        self.outputs.stagnation_pressure=Pt_out
        self.outputs.stagnation_enthalpy=ht_out
        self.outputs.fuel_to_air_ratio = f #0.0215328 #f
    
    
    
    __call__ = compute
