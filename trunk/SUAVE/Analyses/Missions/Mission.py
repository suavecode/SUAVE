""" Mission.py: Top-level mission class """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Structure import Data, Data_Exception
from SUAVE.Structure import Container as ContainerBase
from Segments import Segment

# ----------------------------------------------------------------------
#   Class
# ----------------------------------------------------------------------

class Mission(Data):
    """ Mission.py: Top-level mission class """
    
    def __defaults__(self):
        
        self.tag = 'Mission'
        
        self.segments = Segment.Container()
    
    def append_segment(self,segment):
        """ Add a Mission Segment  """
        self.segments.append(segment)
        return
    
    def evaluate(self,interface):
        
        ## TODO
        # evaluate_mission()
        
        return Data()
    
    
    def merge_conditions(self):
        
        import numpy as np
        
        # merge all segment conditions
        def stack_condition(a,b):
            if isinstance(a,np.ndarray):
                return np.vstack([a,b])
            else:
                return None
    
        conditions = None
        for segment in self.segments:
            if conditions is None:
                conditions = segment.conditions
                continue
            conditions = conditions.do_recursive(stack_condition,segment.conditions)    
        
        return conditions

# ----------------------------------------------------------------------
#   Cotnainer Class
# ----------------------------------------------------------------------

class Container(ContainerBase):
    
    def evaluate(self,interface):
        results = SUAVE.Analyses.Results()
        
        for mission in self:
            result = mission.evaluate(interface)
            results[mission.tag] = result
            
        return results



# Link container
Mission.Container = Container
