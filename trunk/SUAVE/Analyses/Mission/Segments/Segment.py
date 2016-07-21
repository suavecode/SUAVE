# Segment.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

from SUAVE.Analyses import Analysis, Settings, Process
from Conditions import State

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

class Segment(Analysis):
    
    def __defaults__(self):
        
        self.settings = Settings()
        
        self.state = State()

        self.analyses = Analysis.Container()
        
        self.process = Process()
        self.process.initialize         = Process()
        self.process.converge           = Process()
        self.process.iterate            = Process()
        self.process.iterate.unknowns   = Process()
        self.process.iterate.initials   = Process()
        self.process.iterate.conditions = Process()
        self.process.iterate.residuals  = Process()
        self.process.finalize           = Process()
        
        return
        

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
            state = self.state
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