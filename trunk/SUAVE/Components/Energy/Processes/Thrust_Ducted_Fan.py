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

class Thrust_Ducted_Fan(Energy_Component):
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
        f                    = self.inputs.normalized_fuel_flow_rate
        
        u8 = self.inputs.fan_exhaust_flow_speed
        
        Fsp = (u8-u0)/a0
        try:
            Isp = Fsp/f*a0/g
        except ZeroDivisionError:
            Isp = np.inf
        sfc = 3600./Isp
        
        self.outputs.specific_thrust           = Fsp
        self.outputs.specific_impulse          = Isp
        self.outputs.specific_fuel_consumption = sfc
    
    __call__ = compute         

