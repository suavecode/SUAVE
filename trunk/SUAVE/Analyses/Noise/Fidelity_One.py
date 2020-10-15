## @ingroup Analyses-Noise
# Fidelity_One.py
#
# Created:  
# Modified: Feb 2016, A. Wendorff
# Modified: Oct 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE 
from SUAVE.Core import Data
from .Noise     import Noise 

# noise imports 
from SUAVE.Methods.Noise.Fidelity_One.Airframe    import noise_airframe_Fink
from SUAVE.Methods.Noise.Fidelity_One.Engine      import noise_SAE 
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import pnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_tone_correction
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import epnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_certification_limits
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_geometric
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_counterplot

from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_low_fidelity import propeller_low_fidelity
from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_noise_sae    import propeller_noise_sae
from SUAVE.Methods.Noise.Fidelity_One.compute_total_aircraft_noise     import compute_total_aircraft_noise

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
## @ingroup Analyses-Noise
class Fidelity_One(Noise):
    
    """ SUAVE.Analyses.Noise.Fidelity_One()
    
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
        self.harmonics        = np.empty(shape=[0, 1])
      
        settings = self.settings
        settings.flyover        = 0     
        settings.approach       = 0
        settings.sideline       = 0
        settings.mic_x_position = 0 
        
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
    
        # unpack
        harmonics     = self.harmonics
        configuration = self.configuration # to be used in the future 
        geometry      = self.geometry      # to be used in the future 
        
        # Compute Propeller Noise 
        #propeller_low_fidelity(conditions,harmonics) 
        
        # Compute Fan Noise 
        
        # Compute Core Noise 
        
        # Compute Airframe Noise 
        
        # Sum Noise Sources   
        #total_noise = compute_total_aircraft_noise(conditions)
        
        #conditions.noise.total = total_noise
        # Noise 
        #self.sideline_init     = noise_sideline_init
        #self.takeoff_init      = noise_takeoff_init
        #self.sideline          = noise_sideline
        #self.flyover           = noise_flyover
        #self.approach          = noise_approach 
        
    
        #turbofan = config.propulsors['turbofan']
        
        #engine_flag       = config.engine_flag  #remove engine noise component from the approach segment
        
        #geometric      = noise_geometric(noise_segment,analyses,config)
        
        #airframe_noise = noise_airframe_Fink(config,analyses,noise_segment)  
    
        #engine_noise   = noise_SAE(turbofan,noise_segment,config,analyses) 
        
         
        return   

