## @ingroup Components-Configs
# Config.py
#
# Created:  Oct 2014, T. Lukacyzk
# Modified: Jan 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core    import Diffed_Data, Data
from SUAVE.Vehicle import Vehicle

from copy import deepcopy

# ----------------------------------------------------------------------
#  Config
# ----------------------------------------------------------------------

## @ingroup Components-Configs
class Config(Diffed_Data):
    """ SUAVE.Components.Config()
    
        The Top Level Configuration Class
        
            Assumptions:
            None
            
            Source:
            N/A
    """
    
    def __defaults__(self):
        """ This sets the default values for the configuration.
        
                Assumptions:
                None
                
                Source:
                N/A
                
                Inputs:
                None
                
                Outputs:
                None
                
                Properties Used:
                N/A
        """
        self.tag    = 'config'
        self._base  = Vehicle()
        self._diff  = Data()        
        
        
    def __init__(self,base=None):
        """ Initializes the new Diffed_Data() class through a deepcopy
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """  
        if base is None: base = Vehicle()
        self._base = base
        this = deepcopy(base) # deepcopy is needed here to build configs - Feb 2016, T. MacDonald
        Vehicle.__init__(self,this)        