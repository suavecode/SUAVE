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
    
class Motor(Energy_Component):
    
    def __defaults__(self):
        
        self.resistance         = 0.0
        self.no_load_current    = 0.0
        self.speed_constant     = 0.0
        self.propeller_radius   = 0.0
        self.propeller_Cp       = 0.0
        self.gear_ratio         = 0.0
        self.gearbox_efficiency = 0.0
        self.expected_current   = 0.0
    
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
                Cp is not a function of rpm or RE
               
        """
        #Unpack
        V     = conditions.freestream.velocity[:,0]
        rho   = conditions.freestream.density[:,0]
        Res   = self.resistance
        etaG  = self.gearbox_efficiency
        exp_i = self.expected_current
        io    = self.no_load_current + exp_i*(1-etaG)
        G     = self.gear_ratio
        Kv    = self.speed_constant/G
        R     = self.propeller_radius
        Cp    = self.propeller_Cp 
        v     = self.inputs.voltage

        #Omega
        #This is solved by setting the torque of the motor equal to the torque of the prop
        #It assumes that the Cp is constant
        omega1  =   (np.pi**(3./2.)*((- 16.*Cp*io*rho*(Kv**3.)*(R**5.)*(Res**2.) +
                    16.*Cp*rho*v*(Kv**3.)*(R**5.)*Res + (np.pi**3.))**(0.5) - 
                    np.pi**(3./2.)))/(8.*Cp*(Kv**2.)*(R**5.)*Res*rho)

        # store to outputs
        self.outputs.omega = omega1
        
        #Q = ((v-omega1/Kv)/Res -io)/Kv
        #P = Q*omega1
        
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
        #etam=(1-io/i)*(1-i*Res/v)
        
        return i