#motor.py
# 
# Created:  Emilio Botero, Jun 2014
# Modified:  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
from SUAVE.Attributes import Units
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Motor Class
# ----------------------------------------------------------------------
    
class Motor_Lo_Fid(Energy_Component):
    
    def __defaults__(self):
        
        self.resistance         = 0.0
        self.no_load_current    = 0.0
        self.speed_constant     = 0.0
        self.gear_ratio         = 0.0
        self.gearbox_efficiency = 0.0
        self.expected_current   = 0.0
        self.motor_efficiency   = 0.0
    
    def omega(self,conditions):
        """ The motor's rotation rate
            
            Inputs:
                Motor resistance - in ohms
                Motor zeros load current - in amps
                Motor Kv - in rad/s/volt
                Propeller radius - in meters
                Propeller Cp - power coefficient
                Freestream velocity - m/s
                Freestream dynamic pressure - kg/m/s^2
                
            Outputs:
                The motor's rotation rate
               
            Assumptions:
                The motor operates at a given efficiency
               
        """
        #Unpack
        V     = conditions.freestream.velocity[:,0,None]
        rho   = conditions.freestream.density[:,0,None]
        Res   = self.resistance
        etaG  = self.gearbox_efficiency
        exp_i = self.expected_current
        io    = self.no_load_current + exp_i*(1-etaG)
        G     = self.gear_ratio
        Kv    = self.speed_constant/G
        etam  = self.motor_efficiency
        v     = self.inputs.voltage
        

        #Omega
        omega1 = (Kv*v)/2 + (Kv*(Res**2*io**2 - 2*Res*etam*io*v - 2*Res*io*v + etam**2*v**2 - 2*etam*v**2 + v**2)**(1/2))/2 - (Kv*Res*io)/2 + (Kv*etam*v)/2
        Q = ((v-omega1/Kv)/Res -io)/Kv

        # store to outputs
        self.outputs.omega  = omega1
        self.outputs.torque = Q
        
        return omega1
    
    def current(self,conditions):
        """ The motor's current
            
            Inputs:
                Motor resistance - in ohms
                Motor Kv - in rad/s/volt
                Voltage - volts
                Gear ratio - ~
                Rotation rate - rad/s
                
            Outputs:
                The motor's current
               
            Assumptions:
                Cp is invariant
               
        """    
        
        # Unpack
        G    = self.gear_ratio
        Kv   = self.speed_constant
        Res  = self.resistance
        v    = self.inputs.voltage
        omeg = self.omega(conditions)*G
        etaG = self.gearbox_efficiency
        exp_i = self.expected_current
        io    = self.no_load_current + exp_i*(1-etaG)
        
        i=(v-omeg/Kv)/Res
        
        # This line means the motor cannot recharge the battery
        i[i < 0.0] = 0.0

        # Pack
        self.outputs.current = i
        
        #Q = (i-io)/Kv
        #print(i*v)
        #pshaft= (i-io)*(v-i*Res)   
        etam=(1-io/i)*(1-i*Res/v)
        conditions.propulsion.etam = etam
        #print etam
        
        return i