#esc.py
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
from copy import deepcopy
# ----------------------------------------------------------------------
#  Electronic Speed Controller Class
# ----------------------------------------------------------------------

class ESC(Energy_Component):
    
    def __defaults__(self):
        
        self.eff = 0.0
    
    def voltageout(self,conditions):
        """ The electronic speed controllers voltage out
            
            Inputs:
                eta - [0-1] throttle setting
                self.inputs.voltage() - a function that returns volts into the ESC
               
            Outputs:
                voltage out of the ESC
               
            Assumptions:
                The ESC's output voltage is linearly related to throttle setting
               
        """
        # Unpack, deep copy since I replace values
        eta = deepcopy(conditions.propulsion.throttle[:,0])
        
        # Negative throttle is bad
        eta[eta<=0.0] = 0.0
        
        # Cap the throttle
        eta[eta>=1.1] = 1.1
        
        voltsin  = self.inputs.voltagein
        voltsout = eta*voltsin
        
        # Pack the output
        self.outputs.voltageout = voltsout
        
        return voltsout
    
    def currentin(self):
        """ The current going in
            
            Inputs:
                eff - [0-1] efficiency of the ESC
                self.inputs.power() - a function that returns power
               
            Outputs:
                Current into the ESC
               
            Assumptions:
                The ESC draws current.
               
        """
        
        # Unpack
        eff        = self.eff
        currentout = self.inputs.currentout
        
        currentin  = currentout/eff
        
        # Pack
        self.outputs.currentin = currentin
        
        return currentin