""" Internal_Combustion.py: internal combustion (piston-cylinder) engine class """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data
from Propulsor import Propulsor
# from SUAVE.Methods.Power import RunBattery
import numpy as np

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------

class Motor_Bat(Propulsor):   #take simple IC engine and attach a motor to it
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
        battery = config.Energy.Storages['Battery']
       

        F = self.F_min_static + (self.F_max_static - self.F_min_static)*eta
        #mdot = mdot_min  + (mdot_max - mdot_min)*eta
        
        #now include fuel cell
        Ptotal=segment.V*F
   
        #print segment.V  
        CF = F/(segment.q*np.pi*(self.D/2)**2)
        
       

        #Isp = F/(mdot*segment.g0)
        eta_pe=.95
        return CF, 0.0, eta_pe