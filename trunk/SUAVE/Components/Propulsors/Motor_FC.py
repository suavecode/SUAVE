""" Internal_Combustion.py: internal combustion (piston-cylinder) engine class """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data
# from SUAVE.Methods.Power import RunFuelCell
from Propulsor import Propulsor
import copy
import numpy as np

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Motor_FC(Propulsor):   #take simple engine module and power it with a fuel cell
    def __defaults__(self):
        self.tag = 'Electric Motor with Fuel Cell'
        self.D = 0.0                    # prop diameter (m)
        self.F_min_static = 0.0         # static thrust corresponding to min throttle
        self.F_max_static = 0.0         # static thrust corresponding to max throttle
        self.mdot_min_static = 0.0      # mass flow corresponding to min throttle
        self.mdot_max_static = 0.0      # mass flow corresponding to max throttle

    def __call__(self,eta,segment):

        # unpack fuel cell
        config   = segment.config
        fuel_cell = config.Energy.Converters['Fuel_Cell']
        

        F = self.F_min_static + (self.F_max_static - self.F_min_static)*eta
        #mdot = mdot_min  + (mdot_max - mdot_min)*eta
        
        Preqfc=abs(copy.copy(segment.V*F))
   
                
   
        #print segment.V  
        CF = F/(segment.q*np.pi*(self.D/2)**2)
        mdot1 = fuel_cell(Preqfc)
        mdot=mdot1[0] #reassign to np array

        Isp = F/(mdot*segment.g0)
        #eta_Pe=F*segment.V/Preqfc
        eta_Pe=0
        return CF, Isp, eta_Pe