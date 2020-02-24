## @ingroup Components-Energy-Distributors
# Basic_Power_Electronics.py
#
# Created:  Feb 2020, K.Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Basic Power Electronics Class
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Distributors
class Basic_Power_Electronics(Energy_Component):
    
    def __defaults__(self):
        """ This sets the default values.
    
            Assumptions:
            The efficiency and mass of this component scales only on the delivered power, i.e. the particular voltage and current required is not considered.
    
            Source:
            www.siemens.com/press/en/feature/2015/corporate/2015-03-electromotor.php?
            (Link is now dead, possibly due to RR acquisition of this SiC solid state electronics technolgy)
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
            """         
        
        self.efficiency = 0.0
    
    def power_in(self,conditions):
        """ The input power of the power electronics
        
            Assumptions:
            Efficiency is constant regardless of all conditions, including both power and environmental conditions.
    
            Source:
            N/A
    
            Inputs:
            conditions.propulsion.throttle [0-1] 
            self.inputs.voltage            [volts]
    
            Outputs:
            voltsout                       [volts]
            self.outputs.voltageout        [volts]
    
            Properties Used:
            None
        """

    def voltageout(self,conditions):
        """ The voltage out of the electronic speed controller
        
            Assumptions:
            The ESC's output voltage is linearly related to throttle setting
    
            Source:
            N/A
    
            Inputs:
            conditions.propulsion.throttle [0-1] 
            self.inputs.voltage            [volts]
    
            Outputs:
            voltsout                       [volts]
            self.outputs.voltageout        [volts]
    
            Properties Used:
            None
           
        """
        # Unpack, don't modify the throttle
        eta = (conditions.propulsion.throttle[:,0,None])*1.0
        
        # Negative throttle is bad
        eta[eta<=0.0] = 0.0
        
        # Cap the throttle
        eta[eta>=1.0] = 1.0
        
        voltsin  = self.inputs.voltagein
        voltsout = eta*voltsin
        
        # Pack the output
        self.outputs.voltageout = voltsout
        
        return voltsout
    
    def currentin(self,conditions):
        """ The current going into the speed controller
        
            Assumptions:
                The ESC draws current.
            
            Inputs:
                self.inputs.currentout [amps]
               
            Outputs:
                outputs.currentin      [amps]
            
            Properties Used:
                self.efficiency - [0-1] efficiency of the ESC
               
        """
        
        # Unpack, don't modify the throttle
        eta = (conditions.propulsion.throttle[:,0,None])*1.0        
        eff        = self.efficiency
        currentout = self.inputs.currentout
        currentin  = currentout*eta/eff
        
        # Pack
        self.outputs.currentin = currentin
        self.outputs.power_in  = self.outputs.voltageout*currentin
        
        return currentin