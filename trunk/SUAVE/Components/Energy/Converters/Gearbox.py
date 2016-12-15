# Gearbox.py
#
# Created:  Aug 2016, C. Ilario
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
from SUAVE.Core import Units
from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Gearbox Class
# ----------------------------------------------------------------------

class Gearbox(Energy_Component):
    
    def __defaults__(self):
        
        self.tag = 'GearBox'
        
        self.gearwheel_radius1 = 0. # radius of gearwheel 1
        self.gearwheel_radius2 = 0. # radius of gearwheel 2
        self.efficiency        = 0. # gearbox efficiency
        
        self.inputs.torque     = 0. # input torque
        self.inputs.speed      = 0. # input speed
        self.inputs.power      = 0. # input power
    
    def compute(self):
        
        # unpack the values
        R1  = self.gearwheel_radius1
        R2  = self.gearwheel_radius2
        eta = self.efficiency        
        
        # unpacking the values form inputs
        T1  = self.inputs.torque
        w1  = self.inputs.speed
        P1  = self.inputs.power

        # method to compute gearbox properties

        w2 = w1 * R1/R2 #* eta    # gear output speed
        T2 = T1 * R1/R2 * eta    # gear output torque
        P2 = P1 * eta            # gear output horsepower
        
        # pack computed quantities into outputs
        self.outputs.speed  = w2
        self.outputs.torque = T2
        self.outputs.power  = P2
    
    
    
    __call__ = compute     
    