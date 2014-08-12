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
        
        self.Res        = 0.0
        self.io         = 0.0
        self.kv         = 0.0
        self.propradius = 0.0
        self.propCp     = 0.0
    
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
        Res   = self.Res
        etaG  = self.etaG
        exp_i = self.exp_i
        io    = self.io + exp_i*(1-etaG)
        G     = self.G
        Kv    = self.kv/G
        R     = self.propradius
        Cp    = self.propCp
        v     = self.inputs.voltage[:,0]

        #Omega
        #This is solved by setting the torque of the motor equal to the torque of the prop
        #It assumes that the Cp is constant
        omega1  =   (np.pi**(3./2.)*((- 16.*Cp*io*rho*(Kv**3.)*(R**5.)*(Res**2.) + 16.*Cp*rho*v*(Kv**3.)*(R**5.)*Res + (np.pi**3.))**(0.5) - np.pi**(3./2.)))/(8.*Cp*(Kv**2.)*(R**5.)*Res*rho)

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
        
        G    = self.G
        Kv   = self.kv
        Res  = self.Res
        v    = self.inputs.voltage[:,0]
        etaG = self.etaG
        omeg = self.omega(conditions)*G
        #io   = self.io + self.exp_i*(1-etaG)        
        
        i=(v-omeg/Kv)/Res

        # store to outputs
        self.outputs.current = i
        
        #Q = (i-io)/Kv
        #print(i*v)
        #pshaft= (i-io)*(v-i*Res)   
        #etam=(1-io/i)*(1-i*Res/v)
        
        return i