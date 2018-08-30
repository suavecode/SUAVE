## @ingroup Analyses-Aerodynamics
# Process_Geometry.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Analyses import Process

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
## @ingroup Analyses-Aerodynamics
class Process_Geometry(Process):
    """A process for evaluate over a component group.

    Assumptions:
    None

    Source:
    N/A
    """      
    
    geometry_key = None
    
    def __init__(self,geometry_key):
        """Sets the geometry key for this process.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        geometry_key      <string>

        Outputs:
        None

        Properties Used:
        self.geometry_key <string>
        """          
        self.geometry_key = geometry_key
    
    def evaluate(self,state,settings,geometry):
        """Evaluates preset processes for each component.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        state     (passed to an evaluation function)
        setting   (passed to an evaluation function)
        geometry  (used to get keys and passed to an evaluation function)

        Outputs:
        None

        Properties Used:
        self.geometry_key <string>
        """             
        geometry_items = geometry.deep_get(self.geometry_key)
        
        results = Data()
        
        for key, this_geometry in geometry_items.items():
            result = Process.evaluate(self,state,settings,this_geometry)
            results[key] = result
            
        return results
        
        
        
        