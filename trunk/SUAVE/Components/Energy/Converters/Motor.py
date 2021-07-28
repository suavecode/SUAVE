## @ingroup Components-Energy-Converters
# Motor.py
#
# Created:  Jun 2014, E. Botero
# Modified: Jan 2016, T. MacDonald 
#           Mar 2020, M. Clarke
#           Sep 2020, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Motor Class
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class Motor(Energy_Component):
    """This is a motor component.
    
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
        self.tag                = 'motor'
        self.resistance         = 0.0
        self.no_load_current    = 0.0
        self.speed_constant     = 0.0
        self.propeller_radius   = 0.0
        self.propeller_Cp       = 0.0
        self.efficiency         = 1.0
        self.gear_ratio         = 1.0
        self.gearbox_efficiency = 1.0
        self.expected_current   = 0.0
        self.interpolated_func  = None
    
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
          propulsion.propeller_power_coefficient [-]
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
          propeller_radius                       [m]
        """           
        # Unpack
        V     = conditions.freestream.velocity[:,0,None]
        rho   = conditions.freestream.density[:,0,None]
        Res   = self.resistance
        etaG  = self.gearbox_efficiency
        exp_i = self.expected_current
        io    = self.no_load_current + exp_i*(1-etaG)
        G     = self.gear_ratio
        Kv    = self.speed_constant/G
        R     = self.propeller_radius
        v     = self.inputs.voltage
        Cp    = self.inputs.propeller_CP
        
    
        # Omega
        # This is solved by setting the torque of the motor equal to the torque of the prop
        # It assumes that the Cp is constant
        omega1  =   ((np.pi**(3./2.))*((- 16.*Cp*io*rho*(Kv*Kv*Kv)*(R*R*R*R*R)*(Res*Res) +
                    16.*Cp*rho*v*(Kv*Kv*Kv)*(R*R*R*R*R)*Res + (np.pi*np.pi*np.pi))**(0.5) - 
                    np.pi**(3./2.)))/(8.*Cp*(Kv*Kv)*(R*R*R*R*R)*Res*rho)
        omega1[np.isnan(omega1)] = 0.0
        
        Q = ((v-omega1/Kv)/Res -io)/Kv
        # store to outputs
       
        #P = Q*omega1
        
        self.outputs.torque = Q
        self.outputs.omega = omega1

        return omega1
    
    def torque(self,conditions): 
        """Calculates the motor's torque

        Assumptions:

        Source:
        N/A

        Inputs:

        Outputs:
        self.outputs.torque    [N-m] 

        Properties Used:
        self.
          gear_ratio           [-]
          speed_constant       [radian/s/V]
          resistance           [ohm]
          outputs.omega        [radian/s]
          gearbox_efficiency   [-]
          expected_current     [A]
          no_load_current      [A]
          inputs.volage        [V]
        """
        
        Res   = self.resistance
        etaG  = self.gearbox_efficiency
        exp_i = self.expected_current
        io    = self.no_load_current + exp_i*(1-etaG)
        G     = self.gear_ratio
        Kv    = self.speed_constant/G
        v     = self.inputs.voltage
        omega = self.inputs.omega
        
        # Torque
        Q = ((v-omega/Kv)/Res -io)/Kv
        
        self.outputs.torque = Q
        self.outputs.omega  = omega
    
        return Q
    
    def voltage_current(self,conditions):
        """Calculates the motor's voltage and current

        Assumptions:

        Source:
        N/A

        Inputs:

        Outputs:
        self.outputs.current   [A]
        conditions.
          propulsion.volage    [V]
        conditions.
          propulsion.etam      [-] 

        Properties Used:
        self.
          gear_ratio           [-]
          speed_constant       [radian/s/V]
          resistance           [ohm]
          outputs.omega        [radian/s]
          gearbox_efficiency   [-]
          expected_current     [A]
          no_load_current      [A]
        """                      
               
        Res   = self.resistance
        etaG  = self.gearbox_efficiency
        exp_i = self.expected_current
        io    = self.no_load_current + exp_i*(1-etaG)
        G     = self.gear_ratio
        kv    = self.speed_constant/G
        Q     = self.inputs.torque
        omega = self.inputs.omega        
        
        v = (Q*kv+io)*Res + omega/kv
        i = (v-omega/kv)/Res
        
        self.outputs.voltage = v
        self.outputs.current = i
        
        etam=(1-io/i)*(1-i*Res/v)
        conditions.propulsion.etam = etam        
        
        return
    
    
    def current(self,conditions):
        """Calculates the motor's current

        Assumptions:

        Source:
        N/A

        Inputs:
        self.inputs.voltage    [V]

        Outputs:
        self.outputs.current   [A]
        conditions.
          propulsion.etam      [-] 

        Properties Used:
        self.
          gear_ratio           [-]
          speed_constant       [radian/s/V]
          resistance           [ohm]
          outputs.omega        [radian/s]
          gearbox_efficiency   [-]
          expected_current     [A]
          no_load_current      [A]
        """                      
        
        # Unpack
        G     = self.gear_ratio
        Kv    = self.speed_constant
        Res   = self.resistance
        v     = self.inputs.voltage
        omeg  = self.outputs.omega*G
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
        
        return i, etam