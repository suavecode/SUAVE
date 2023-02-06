## @ingroup Components-Energy-Converters
# Lift_Rotor.py
#
# Created:  July 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from MARC.Core import Data
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

        self.tag                              = 'lift_rotor'
        self.orientation_euler_angles         = [0.,np.pi/2.,0.] # This is Z-direction thrust up in vehicle frame
        self.use_2d_analysis                  = False
        self.variable_pitch                   = False 

        self.hover                            = Data()    
        self.hover.design_thrust              = None
        self.hover.design_torque              = None
        self.hover.design_power               = None
        self.hover.design_angular_velocity    = None
        self.hover.design_tip_mach            = None
        self.hover.design_freestream_velocity = None
        self.hover.design_acoustics           = None
        self.hover.design_performance         = None
        self.hover.design_SPL_dBA             = None
        self.hover.design_Cl                  = None
        self.hover.design_thrust_coefficient  = None
        self.hover.design_power_coefficient   = None 
        
        self.oei                              = Data()
        self.oei.design_thrust                = None
        self.oei.design_torque                = None
        self.oei.design_power                 = None
        self.oei.design_angular_velocity      = None
        self.oei.design_freestream_velocity   = None
        self.oei.design_tip_mach              = None  
        self.oei.design_altitude              = None
        self.oei.design_acoustics             = None
        self.oei.design_performance           = None
        self.oei.design_SPL_dBA               = None
        self.oei.design_Cl                    = None
        self.oei.design_thrust_coefficient    = None
        self.oei.design_power_coefficient     = None         
         