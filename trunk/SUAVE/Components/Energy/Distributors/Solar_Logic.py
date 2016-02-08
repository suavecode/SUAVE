# Solar_Logic.py
#
# Created:  Jun 2014, E. Botero
# Modified: Jan 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Solar Logic Class
# ----------------------------------------------------------------------
    
class Solar_Logic(Energy_Component):
    
    def __defaults__(self):
        
        self.MPPT_efficiency = 0.0
        self.system_voltage  = 0.0
    
    def voltage(self):
        """ The system voltage
            
            Inputs:
                voltage
               
            Outputs:
                voltage
               
            Assumptions:
                this function practically does nothing
               
        """
        volts = self.system_voltage
        
        self.outputs.system_voltage = volts
        
        return volts

    def logic(self,conditions,numerics):
        """ The power being sent to the battery
            
            Inputs:
                payload power
                avionics power
                current to the esc
                voltage of the system
                MPPT efficiency
               
            Outputs:
                power to the battery
               
            Assumptions:
                Many: the system voltage is constant, the maximum power
                point is at a constant voltage
               
        """
        #Unpack
        pin         = self.inputs.powerin[:,0,None]
        pavionics   = self.inputs.pavionics
        ppayload    = self.inputs.ppayload
        volts_motor = self.inputs.volts_motor
        volts       = self.voltage()
        esccurrent  = self.inputs.currentesc
        I           = numerics.time.integrate
        
        pavail = pin*self.MPPT_efficiency
        
        plevel = pavail -pavionics -ppayload - volts_motor*esccurrent
        
        # Integrate the plevel over time to assess the energy consumption
        # or energy storage
        e = np.dot(I,plevel)
        
        # Send or take power out of the battery, Pack up
        self.outputs.current         = (plevel/volts)
        self.outputs.power_in        = plevel
        self.outputs.energy_transfer = e
        
        
        return 