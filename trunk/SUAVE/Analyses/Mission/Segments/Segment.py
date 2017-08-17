## @ingroup Analyses-Mission-Segments
# Segment.py
#
# Created:  
# Modified: Sep 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports

from SUAVE.Analyses import Analysis, Settings, Process
from Conditions import State

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
        
        return
        

    def initialize(self,state):
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
        self.process.initialize(self,state)
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
    
    def iterate(self,state):
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
        self.process.iterate(self,state)
        return
    
    def finalize(self,state):
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
        self.process.finalize(self,state)
        return
 
    def compile(self):
        """ This does nothing
    
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
        self.process(self,state)
        return state
    
    
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