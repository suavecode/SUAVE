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

        self.tag                         = 'prop_rotor'
        self.orientation_euler_angles    = [0.,0.,0.] # This is X-direction thrust in vehicle frame
        self.use_2d_analysis             = False       
        self.variable_pitch              = True
        self.design_thrust_cruis         = None     
        self.design_thrust_hover         = None
        self.design_power_cruise         = None 
        self.design_power_hover          = None
        self.inputs.pitch_command_hover  = 0.0      
        self.inputs.pitch_command_cruise = 0.0
        

        self.optimization_parameters                                   = Data() 
        self.optimization_parameters.slack_constaint                   = 1E-2 # slack constraint 
        self.optimization_parameters.ideal_SPL_dBA                     = 45 
        self.optimization_parameters.aeroacoustic_weight               = 1.   # 1 = aerodynamic optimization, 0.5 = equally weighted aeroacoustic optimization, 0 = acoustic optimization     
        self.optimization_parameters.multiobjective_performance_weight = 0.5 
        self.optimization_parameters.multiobjective_acoustic_weight    = 0.5  
        
