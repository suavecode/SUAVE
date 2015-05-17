
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np
from copy import deepcopy

# SUAVE imports
from SUAVE.Core import Data, Data_Exception

from SUAVE.Analyses import Analysis, Settings, Process

from Conditions import State, Conditions

from SUAVE.Plugins.VyPy.tools import array_type



# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

class Segment(Analysis):
    
    def __defaults__(self):
        
        self.settings = Settings()
        
        self.state = State()

        self.analyses = Analysis.Container()
        
        self.process = Process()
        self.process.initialize       = Process()
        self.process.converge         = Process()
        self.process.iterate          = Process()
        self.process.iterate.initials = Process()
        self.process.finalize         = Process()
        

    def initialize(self,state):
        self.process.initialize(self,state)
        return
    
    def converge(self,state):
        self.process.converge(self,state)    
    
    def iterate(self,state):
        self.process.iterate(self,state)
        return
    
    def finalize(self,state):
        self.process.finalize(self,state)
        return
 
    def compile(self):
        return
    
                        
    def evaluate(self,state=None):
        if state is None:
            state = deepcopy(self.state)
        self.process(self,state)
        return state
    
    
# ----------------------------------------------------------------------
#  Container
# ----------------------------------------------------------------------

class Container(Segment):
    
    def __defaults__(self):
                
        self.segments = Process()
        
        self.state = State.Container()
        
    def append_segment(self,segment):
        """ Add a SubSegment  """
        self.segments.append(segment)
        return    
        
Segment.Container = Container