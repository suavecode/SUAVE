## @ingroup Components-Energy-Distributors
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

## @ingroup Components-Energy-Distributors
class Solar_Logic(Energy_Component):
    """ The distributor is a component unique to a solar aircraft. This controls the flow of energy in to and from the battery.
        This includes the basic logic of the maximum power point tracker that modifies the voltage of the panels to
        extract maximum power.
    
        Assumptions:
        None
        
        Source:
        None
    """
    
    
    def __defaults__(self):
        """ This sets the default values.
    
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
        
        self.MPPT_efficiency = 0.0
        self.system_voltage  = 0.0
    
    def voltage(self):
        """ The system voltage
        
            Assumptions:
                this function practically does nothing
                    
            Source:
            N/A
            
            Inputs:
                self.system_voltage         [volts]
               
            Outputs:
                self.outputs.system_voltage [volts]
                
            Properties Used:
            None               
        """
        volts = self.system_voltage
        
        self.outputs.system_voltage = volts
        
        return volts

    def logic(self,conditions,numerics):
        """ The power being sent to the battery
        
            Assumptions:
                the system voltage is constant
                the maximum power point is at a constant voltage
                
            Source:
            N/A
            
            Inputs:
                self.inputs:
                    powerin
                    pavionics
                    ppayload
                    volts_motor
                    currentesc
                numerics.time.integrate

            Outputs:
                self.outputs:
                    current
                    power_in
                    energy_transfer
                    
            Properties Used:
                self.MPPT_efficiency

        """
        #Unpack
        pin         = self.inputs.powerin[:,0,None]
        pavionics   = self.inputs.pavionics
        ppayload    = self.inputs.ppayload
        volts_motor = self.inputs.volts_motor
        esccurrent  = self.inputs.currentesc
        volts       = self.voltage()
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