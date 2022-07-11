## @ingroup Components-Energy-Converters
# Motor_Lo_Fid.py
#
# Created:  Jun 2014, E. Botero
# Modified: Jan 2016, T. MacDonald  
#           Mar 2020, M. Clarke
#           Mar 2020, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# suave imports
import SUAVE

# package imports
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Motor Class
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class Motor_Lo_Fid(Energy_Component):
    """This is a low-fidelity motor component.
    
    Assumptions:
    None

    Source:
    None
    """      
    def __defaults__(self):
        """This sets the default values for the component to function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """              
        self.resistance         = 0.0
        self.no_load_current    = 0.0
        self.speed_constant     = 0.0
        self.gear_ratio         = 0.0
        self.gearbox_efficiency = 0.0
        self.expected_current   = 0.0
        self.motor_efficiency   = 0.0
        self.rated_power        = 0.0
        self.rated_voltage      = 0.0
    
    def omega(self,conditions):
        """Calculates the motor's rotation rate
    
        Assumptions:
        Cp (power coefficient) is constant
    
        Source:
        N/A
    
        Inputs:
        conditions.
          freestream.velocity                    [m/s]
          freestream.density                     [kg/m^3]
        self.inputs.voltage                      [V]
    
        Outputs:
        self.outputs.
          torque                                 [Nm]
          omega                                  [radian/s]
    
        Properties Used:
        self.
          resistance                             [ohms]
          gearbox_efficiency                     [-]
          expected_current                       [A]
          no_load_current                        [A]
          gear_ratio                             [-]
          speed_constant                         [radian/s/V]
        """  
        # Unpack
        Res   = self.resistance
        etaG  = self.gearbox_efficiency
        exp_i = self.expected_current
        io    = self.no_load_current + exp_i*(1-etaG)
        G     = self.gear_ratio
        Kv    = self.speed_constant/G
        etam  = self.motor_efficiency
        v     = self.inputs.voltage
        
        inside = Res*Res*io*io - 2.*Res*etam*io*v - 2.*Res*io*v + etam*etam*v*v - 2.*etam*v*v + v*v
        
        inside[inside<0.] = 0.
        
        # Omega
        omega1 = (Kv*v)/2. + (Kv*(inside)**(1./2.))/2. - (Kv*Res*io)/2. + (Kv*etam*v)/2.

        # If the voltage supplied is too low this function will NaN. However, that really means it won't spin
        omega1[np.isnan(omega1)] = 0.0
        
        # The torque
        Q      = ((v-omega1/Kv)/Res -io)/Kv
        
        omega1[v==0] = 0.

        # store to outputs
        self.outputs.omega  = omega1
        self.outputs.torque = Q
        
        return omega1
    
    def current(self,conditions):
        """Calculates the motor's rotation rate
    
        Assumptions:
        Cp (power coefficient) is constant
    
        Source:
        N/A
    
        Inputs:
        self.inputs.voltage    [V]
    
        Outputs:
        self.outputs.current   [A]
        conditions.
          propulsion.etam      [-]
        i                      [A]
    
        Properties Used:
        self.
          gear_ratio           [-]
          speed_constant       [radian/s/V]
          resistance           [ohm]
          omega(conditions)    [radian/s] (calls the function)
          gearbox_efficiency   [-]
          expected_current     [A]
          no_load_current      [A]
        """       
        
        # Unpack
        G     = self.gear_ratio
        Kv    = self.speed_constant
        Res   = self.resistance
        v     = self.inputs.voltage
        omeg  = self.omega(conditions)*G
        etaG  = self.gearbox_efficiency
        exp_i = self.expected_current
        io    = self.no_load_current + exp_i*(1-etaG)
        
        i=(v-omeg/Kv)/Res
        
        # This line means the motor cannot recharge the battery
        i[i < 0.0] = 0.0

        # Pack
        self.outputs.current = i
         
        etam=(1-io/i)*(1-i*Res/v)
        conditions.propulsion.etam = etam
        
        return i
    
    def power_lo(self,conditions): 
        """Calculates the motor's power output
    
        Assumptions: 
    
        Source:
        N/A
    
        Inputs: 
        self.
           inputs.voltage         [V] 
    
        Outputs:
        self.outputs.
           outputs.power          [W]
           outputs.current        [A]
           
        Properties Used:
        self.
        
          motor_efficiency        [-] 
          rated_power             [W] 
          rated_voltage           [V] 
        """          
        
        etam   = self.motor_efficiency
        power  = self.rated_power
        v_rate = self.rated_voltage
        v_in   = self.inputs.voltage
        
        power_out = power*etam*v_in/v_rate
    
        
        i = power_out/(etam*v_in)
        
        i[power_out == 0.] = 0.
        
        self.outputs.power   = power_out
        self.outputs.current = i
        
        return power_out, i
    
        
        