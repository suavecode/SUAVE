## @ingroup Components-Energy-Converters
# Gearbox.py
#
# Created:  Aug 2016, C. Ilario
# Modified: Feb 2020, M. Clarke 
#           Mar 2020, M. Clarke

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
## @ingroup Components-Energy-Converters
class Gearbox(Energy_Component):
    """This is a gear box component.
    
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
        self.tag = 'GearBox'
        
        self.gearwheel_radius1 = 0. # radius of gearwheel 1
        self.gearwheel_radius2 = 0. # radius of gearwheel 2
        self.efficiency        = 0. # gearbox efficiency
        
        self.inputs.torque     = 0. # input torque
        self.inputs.speed      = 0. # input speed
        self.inputs.power      = 0. # input power
    
    def compute(self):
        """Computes gearbox values.

        Assumptions:
        None

        Source:
        None

        Inputs:
        see properties used

        Outputs:
        self.outputs.
          rotation_speed    [radian/s]
          torque            [Nm]
          power             [W]

        Properties Used:
        self.
          gearwhell_radius1 [m]
          gearwhell_radius2 [m]
          torque1           [Nm]
          speed_1           [radian/s]
          efficiency        [-]
        """            
        
        # unpack the values
        R1  = self.gearwheel_radius1
        R2  = self.gearwheel_radius2
        eta = self.efficiency        
        
        # unpacking the values form inputs
        T1  = self.inputs.torque
        w1  = self.inputs.speed
        P1  = self.inputs.power

        # method to compute gearbox properties

        w2 = w1 * R1/R2          # gear output speed
        T2 = T1 * R1/R2 * eta    # gear output torque
        P2 = P1 * eta            # gear output horsepower
        
        # pack computed quantities into outputs
        self.outputs.speed  = w2
        self.outputs.torque = T2
        self.outputs.power  = P2 
    
    __call__ = compute     
    