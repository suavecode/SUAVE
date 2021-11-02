## @ingroup Analyses-Mission-Segments
# Segment.py
#
# Created:  
# Modified: Sep 2016, E. Botero
#           Oct 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

from SUAVE.Analyses import Analysis, Settings, Process
from .Conditions import State
import numpy as np
from copy import deepcopy

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments
class Segment(Analysis):
    """ The first basic piece of a mission which each segment will expand upon
    
        Assumptions:
        There's a detailed process flow outline in defaults. A mission must be solved in that order.
        
        Source:
        None
    """    
    
    def __defaults__(self):
        """This sets the default values.
    
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
        
        self.settings = Settings()
        
        self.state = State()

        self.analyses = Analysis.Container()
        
        self.process = Process()
        self.process.initialize            = Process()
        self.process.converge              = Process()
        self.process.iterate               = Process()
        self.process.iterate.unknowns      = Process()
        self.process.iterate.initials      = Process()
        self.process.iterate.conditions    = Process()
        self.process.iterate.residuals     = Process()
        self.process.finalize              = Process()
        self.process.finalize.post_process = Process()
        
        self.conditions = self.state.conditions
        
        return
        

    def initialize(self):
        """ This executes the initialize process
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            State  [Data()]
    
            Outputs:
            None
    
            Properties Used:
            None
        """         
        state = self.state
        self.process.initialize(self)
        return
    
    def converge(self,state):
        """ This executes the converge process
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            State  [Data()]
    
            Outputs:
            None
    
            Properties Used:
            None
        """             
        self.process.converge(self,state)    
    
    def iterate(self):
        """ This executes the iterate process
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            State  [Data()]
    
            Outputs:
            None
    
            Properties Used:
            None
        """        
        self.process.iterate(self)
        return
    
    def finalize(self):
        """ This executes the finalize process
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            State  [Data()]
    
            Outputs:
            None
    
            Properties Used:
            None
        """         
        self.process.finalize(self)
        return
    
                        
    def evaluate(self,state=None):
        """ This executes the entire process
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            State  [Data()]
    
            Outputs:
            State  [Data()]
    
            Properties Used:
            None
        """          
        if state is None:
            state = self.state
        self.process(self)
        return self
    
    
    def merged(self):
        """ Combines the states of multiple segments
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            state_out [State()]
    
            Properties Used:
            None
        """              
        
        state_out = State()
        
        for i,(tag,sub_seg) in enumerate(self.segments.items()):
            sub_state = sub_seg.state
            for key in ['unknowns','conditions','residuals']:
                if i == 0:
                    state_out[key] = deepcopy(sub_state[key]) # Necessary deepcopy: otherwise the next step overwrites this state
                else:
                    state_out[key].append_or_update(sub_state[key])

        return state_out

    
    
# ----------------------------------------------------------------------
#  Container
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments
class Container(Segment):
    """ A container for the segment
    
        Assumptions:
        None
        
        Source:
        None
    """    
    
    def __defaults__(self):
        """This sets the default values.
    
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
                
        self.segments = Process()
        
        self.state = State.Container()
        
    def append_segment(self,segment):
        """ Add a SubSegment
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            segment  [Segment()]
    
            Outputs:
            None
    
            Properties Used:
            None
        """          
        self.segments.append(segment)
        return    
        
Segment.Container = Container