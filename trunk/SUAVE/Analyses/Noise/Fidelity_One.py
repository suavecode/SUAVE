# Fidelity_One.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from Noise import Noise

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
class Fidelity_One(Noise):
    
    def __defaults__(self):
        
        self.tag    = 'fidelity_zero_markup'              
    
        # correction factors
        settings = self.settings
        settings.flyover        = 0     
        settings.approach       = 0
        settings.sideline       = 0
        settings.mic_x_position = 0
        