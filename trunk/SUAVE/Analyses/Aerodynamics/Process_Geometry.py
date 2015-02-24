
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Analyses import Process
from SUAVE.Analyses import Results

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Process_Geometry(Process):
    
    geometry_key = None
    
    def __init__(self,geometry_key):
        self.geometry_key = geometry_key
    
    def evaluate(self,state,settings,geometry):
        
        geometry_items = geometry.deep_get(self.geometry_key)
        
        results = Results()
        
        for key, this_geometry in geometry_items.items():
            result = Process.evaluate(self,state,settings,this_geometry)
            reuslts[key] = result
            
        return results
        
        
        
        