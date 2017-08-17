## @ingroup Components-Energy-Converters
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
        
        self.gearwhell_radius1 = 0.     #radius of gearwheel 1
        self.gearwhell_radius2 = 0.     #radius of gearwheel 2
        self.torque_1 = 0.              #Torque of gearwheel 1
        self.torque_2 = 0.              #Torque of gearwheel 2
        self.speed_1 = 0.               #Speed of gear 1
        self.speed_2 = 0.               #Speed of gear 2
        self.efficiency = 0.            #gearbox efficiency
        self.mass_weight = 0.           #mass weight
        
        
    
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
        
        
        # unpacking the values form inputs
        R1  = self.gearwhell_radius1
        R2  = self.gearwhell_radius2
        T1  = self.torque1
        w1  = self.speed_1
        eta = self.efficiency   

        # method to compute gearbox properties

        w2 = w1 * R1/R2 * eta    #gear output speed
        T2 = T1 * R1/R2 * eta    #gear output torque
        P2 = P1 * eta            #gear output horsepower
        
        # pack computed quantities into outputs
        self.outputs.rotation_speed  = w2
        self.outputs.torque     = T2
        self.outputs.power     = P2
    
    
    
    __call__ = compute     
    