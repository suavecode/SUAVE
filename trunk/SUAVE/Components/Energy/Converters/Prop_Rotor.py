## @ingroup Components-Energy-Converters
# Prop_Rotor.py
#
# Created:  Feb 2021, M. Clarke
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data
from .Rotor import Rotor

# ----------------------------------------------------------------------
#  Propeller Class
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Converters
class Prop_Rotor(Rotor):
    """This is a prop_rotor component, and is a sub-class of rotor.
    
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

        self.tag                                 = 'prop_rotor'
        self.orientation_euler_angles            = [0.,0.,0.] # This is X-direction thrust in vehicle frame
        self.use_2d_analysis                     = False       
        self.variable_pitch                      = True 
        
        self.hover                               = Data()
        self.hover.design_thrust                 = None
        self.hover.design_torque                 = None
        self.hover.design_power                  = None
        self.hover.design_angular_velocity       = None
        self.hover.design_tip_mach               = None
        self.hover.design_acoustics              = None
        self.hover.design_performance            = None
        self.hover.design_freestream_velocity    = None
        self.hover.design_SPL_dBA                = None
        self.hover.design_Cl                     = None
        self.hover.design_thrust_coefficient     = None
        self.hover.design_power_coefficient      = None  
        
        self.oei                                 = Data()   
        self.oei.design_thrust                   = None
        self.oei.design_torque                   = None
        self.oei.design_power                    = None
        self.oei.design_angular_velocity         = None
        self.oei.design_tip_mach                 = None
        self.oei.design_acoustics                = None
        self.oei.design_performance              = None 
        self.oei.design_freestream_velocity      = None   
        self.oei.design_altitude                 = None
        self.oei.design_SPL_dBA                  = None
        self.oei.design_Cl                       = None
        self.oei.design_thrust_coefficient       = None
        self.oei.design_power_coefficient        = None  

        self.cruise                              = Data()     
        self.cruise.design_thrust                = None
        self.cruise.design_torque                = None
        self.cruise.design_power                 = None
        self.cruise.design_angular_velocity      = None
        self.cruise.design_tip_mach              = None
        self.cruise.design_acoustics             = None
        self.cruise.design_performance           = None
        self.cruise.design_SPL_dBA               = None
        self.cruise.design_Cl                    = None
        self.cruise.design_thrust_coefficient    = None
        self.cruise.design_power_coefficient     = None         
