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


class Ram(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Ram
        a Ram class that is used to convert static properties into
        stagnation properties
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        
        self.tag = 'Ram'
        
        
        self.outputs.stagnation_temperature=1.0
        self.outputs.stagnation_pressure =1.0
    
    
    
    
    def compute(self,conditions):
        
        #unpack the variables
        
        print conditions
        Po = conditions.freestream.pressure
        To = conditions.freestream.temperature
        working_fluid = self.inputs.working_fluid
        M = conditions.freestream.mach_number
        
        
        
        #method
        
        
        
        gamma              = 1.4
        Cp                 = 1.4*287.87/(1.4-1)
        R                  = 287.87
        
        #gamma              = working_fluid.compute_gamma(To,Po)
        #Cp                 = working_fluid.compute_cp(To,Po)
        #R                  = (gamma-1)/gamma * Cp
        
        
        ao     =  np.sqrt(Cp/(Cp-R)*R*To)
        
        #Compute the stagnation quantities from the input static quantities
        stagnation_temperature = To*(1+((gamma-1)/2 *M**2))
        
        
        stagnation_pressure = Po* ((1+(gamma-1)/2 *M**2 )**3.5 )
        
        
        
        #pack outputs
        self.outputs.stagnation_temperature =stagnation_temperature
        self.outputs.stagnation_pressure =stagnation_pressure
        self.outputs.gamma              = gamma
        self.outputs.Cp                 = Cp
        self.outputs.R                  = R
        
        conditions.freestream.stagnation_temperature =  stagnation_temperature
        conditions.freestream.stagnation_pressure = stagnation_pressure
        conditions.freestream.gamma              = gamma
        conditions.freestream.Cp                 = Cp
        conditions.freestream.R                  = R
        conditions.freestream.speed_of_sound     = ao
    
    
    
    
    
    __call__ = compute

