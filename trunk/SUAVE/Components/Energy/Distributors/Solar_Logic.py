#Solar_Logic.py
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
from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

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
        I           = numerics.integrate_time
        
        pavail = pin*self.MPPT_efficiency
        
        plevel = pavail -pavionics -ppayload - volts_motor*esccurrent
        
        # Integrate the plevel over time to assess the energy consumption
        # or energy storage
        e = np.dot(I,plevel)
        
        # Send or take power out of the battery, Pack up
        batlogic      = Data()
        batlogic.pbat = plevel
        batlogic.Ibat = (plevel/volts)
        batlogic.e    = e
        
        # Output
        self.outputs.batlogic = batlogic
        
        return 