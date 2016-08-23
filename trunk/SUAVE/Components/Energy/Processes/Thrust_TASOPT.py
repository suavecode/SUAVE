# Thrust.py
#
# Created:  Jul 2014, A. Variyar
# Modified: Feb 2016, T. MacDonald, A. Variyar, M. Vegh


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

import SUAVE

from SUAVE.Core import Units

# package imports
import numpy as np
import scipy as sp


from SUAVE.Core import Data
from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Components.Propulsors.Propulsor import Propulsor


# ----------------------------------------------------------------------
#  Thrust Process
# ----------------------------------------------------------------------

class Thrust_TASOPT(Energy_Component):
    """ SUAVE.Components.Energy.Gas_Turbine.Thrust
        a component that computes the thrust and other output properties
        
        this class is callable, see self.__call__
        
        """
    
    def __defaults__(self):
        
        #setting the default values
        self.tag ='Thrust'
	
 
    def compute(self,conditions):
        
        u0                   = conditions.freestream.velocity
        a0                   = conditions.freestream.speed_of_sound
        g                    = conditions.freestream.gravity
        
        f  = self.inputs.normalized_fuel_flow_rate
        u6 = self.inputs.core_exhaust_flow_speed
        u8 = self.inputs.fan_exhaust_flow_speed
        bypass_ratio = self.inputs.bypass_ratio
        
        Fsp = ((1.+f)*u6 - u0 + bypass_ratio*(u8-u0))/((1.+bypass_ratio)*a0)
        Isp = Fsp/f*a0/g*(1.0+bypass_ratio)
        sfc = 3600./Isp
        
        self.outputs.specific_thrust           = Fsp
        self.outputs.specific_impulse          = Isp
        self.outputs.specific_fuel_consumption = sfc
    
    __call__ = compute         

