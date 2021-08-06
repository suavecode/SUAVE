## @ingroup Components-Energy-Converters
# Lift_Rotor.py
#
# Created:  July 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from .Rotor import Rotor
import numpy as np

# ----------------------------------------------------------------------
#  Lift Rotor Class
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Converters
class Lift_Rotor(Rotor):
    """This is a lift rotor component, and is a sub-class of rotor.
    
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

        self.tag                       = 'lift_rotor'
        self.orientation_euler_angles  = [0.,np.pi/2.,0.] # This is Z-direction thrust up in vehicle frame
        self.use_2d_analysis           = False
        self.variable_pitch            = False
        