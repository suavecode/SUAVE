## @ingroup Analyses-Noise
# Fidelity_Zero.py
#
# Created:   
# Modified: Oct 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE 
from SUAVE.Core import Data
from .Noise     import Noise  
# package imports
import numpy as np

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

        self.configuration    = Data()
        self.geometry         = Data()    
        self.flyover          = 0     
        self.approach         = 0
        self.sideline         = 0
        self.mic_x_position   = 0 
        
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
        self.geometry
        """                          
    
        # unpack
        geometry         = self.geometry      # really a vehicle object
        configuration    = self.configuration  
    
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
     
        
        return   0
