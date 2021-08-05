## @ingroup Analyses-Noise
# Fidelity_Zero.py
#
# Created:   
# Modified: Oct 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE 
from SUAVE.Core import Data , Units
from .Noise     import Noise  

from SUAVE.Methods.Noise.Fidelity_Zero.shevell import shevell

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
## @ingroup Analyses-Noise
class Fidelity_Zero(Noise):
    
    """ SUAVE.Analyses.Noise.Fidelity_Zero()
    
        The Fidelity One Noise Analysis Class
        
            Assumptions:
            None
            
            Source:
            N/A
    """
    
    def __defaults__(self):
        
        """ This sets the default values for the analysis.
        
                Assumptions:
                None
                
                Source:
                N/A
                
                Inputs:
                None
                
                Output:
                None
                
                Properties Used:
                N/A
        """
        
        # Initialize quantities 
        self.geometry                 = Data()    
        self.flyover                  = False     
        self.approach                 = False
        self.sideline                 = False
        return
        
    def finalize(self):
        """Finalizes the surrogate needed for lift calculation.
    
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
     
    
    def evaluate_noise(self,conditions):
        """ Process vehicle to setup geometry, condititon and configuration
    
        Assumptions:
        None
    
        Source:
        N/4
    
        Inputs:
        conditions - DataDict() of aerodynamic conditions
        results    - DataDict() of moment coeffients and stability and body axis derivatives
    
        Outputs:
        None
    
        Properties Used:
        self.geometry
        
        """         
        # unpack 
        geometry = self.geometry   
        
        if 'turbofan' in geometry.networks: 
            weight_landing    = conditions.weights.total_mass[-1,0]
            number_of_engines = geometry.networks['turbofan'].number_of_engines
            thrust_sea_level  = geometry.networks['turbofan'].sealevel_static_thrust * Units.force_pounds
            thrust_landing    = 0.45 * thrust_sea_level
            
            # Run Shevell Correlations  
            outputs = shevell(weight_landing, number_of_engines, thrust_sea_level, thrust_landing) 
            
            self.flyover          = outputs.takeoff  
            self.approach         = outputs.landing
            self.sideline         = outputs.side_line
        
        return    
