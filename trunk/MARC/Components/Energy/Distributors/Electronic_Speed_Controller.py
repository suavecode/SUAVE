## @ingroup Components-Energy-Distributors
# Electronic_Speed_Controller.py
#
# Created:  Jun 2014, E. Botero
# Modified: Jan 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# MARC imports
import MARC

from MARC.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Electronic Speed Controller Class
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Distributors
class Electronic_Speed_Controller(Energy_Component):
    
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
        
        self.efficiency = 0.0
    
    def voltageout(self,eta):
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
        # Negative throttle is bad
        eta[eta<=0.0] = 0.0
        
        # Cap the throttle
        eta[eta>=1.0] = 1.0
        
        voltsin  = self.inputs.voltagein
        voltsout = eta*voltsin
        
        # Pack the output
        self.outputs.voltageout = voltsout
        
        return voltsout
    
    def currentin(self,eta):
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
        eff        = self.efficiency
        currentout = self.inputs.currentout
        currentin  = currentout*eta/eff # The inclusion of eta satisfies a power balance: p_in = p_out/eff
        
        # Pack
        self.outputs.currentin = currentin
        self.outputs.power_in  = self.inputs.voltagein*currentin
        
        return currentin