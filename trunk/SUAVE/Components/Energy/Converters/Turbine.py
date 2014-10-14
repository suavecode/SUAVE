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

class Turbine(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Turbine
        a Turbine component
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        
        self.tag ='Turbine'
        self.mechanical_efficiency =1.0
        self.polytropic_efficiency = 1.0
        
        self.inputs.stagnation_temperature = 1.0
        self.inputs.stagnation_pressure = 1.0
        self.inputs.fuel_to_air_ratio = 1.0
        
        
        self.outputs.stagnation_temperature=1.0
        self.outputs.stagnation_pressure=1.0
        self.outputs.stagnation_enthalpy=1.0
    
    
    
    
    def compute(self,conditions):
        
        #unpack inputs
        gamma=conditions.freestream.gamma
        Cp = conditions.freestream.Cp
        
        Tt_in =self.inputs.stagnation_temperature
        Pt_in =self.inputs.stagnation_pressure
        
        alpha =  self.inputs.bypass_ratio
        f =self.inputs.fuel_to_air_ratio
        compressor_work = self.inputs.compressor.work_done
        fan_work = self.inputs.fan.work_done
        
        eta_mech =  self.mechanical_efficiency
        etapolt =  self.polytropic_efficiency
        
        #Using the stagnation enthalpy drop across the corresponding turbine and the fuel to air ratio to compute the energy drop across the turbine
        deltah_ht=-1/(1+f)*1/eta_mech*((compressor_work)+ alpha*(fan_work))
        
        #Compute the output stagnation quantities from the inputs and the energy drop computed above
        Tt_out=Tt_in+deltah_ht/Cp
        Pt_out=Pt_in*(Tt_out/Tt_in)**(gamma/((gamma-1)*etapolt))
        ht_out=Cp*Tt_out   #h(Tt4_5)
        
        
        
        #pack outputs
        self.outputs.stagnation_temperature=Tt_out
        self.outputs.stagnation_pressure=Pt_out
        self.outputs.stagnation_enthalpy=ht_out
    
    
    __call__ = compute