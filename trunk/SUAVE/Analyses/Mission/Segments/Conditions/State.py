## @ingroup Analyses-Mission-Segments-Conditions
# State.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff
#           Jan 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from .Conditions import Conditions
from .Unknowns   import Unknowns
from .Residuals  import Residuals
from .Numerics   import Numerics

import SUAVE
from SUAVE.Core.Arrays import array_type
from SUAVE.Core import DataOrdered

# ----------------------------------------------------------------------
#  State
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Conditions
class State(Conditions):
    """ Creates the State data structure for storing daata that solved in a mission
    
        Assumptions:
        None
        
        Source:
        None
    """    
    
    
    def __defaults__(self):
        """ This sets the default values.
    
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
        
        self.tag        = 'state'
        self.unknowns   = Unknowns()
        self.conditions = Conditions()
        self.residuals  = Residuals()
        self.numerics   = Numerics()
        self.initials   = Conditions()
        
    def expand_rows(self,rows):
        """ Makes a 1-D array the right size. Often used after a mission is initialized to size out the vectors to the
            right size.
        
            Assumptions:
            Doesn't expand initials or numerics
    
            Source:
            N/A
    
            Inputs:
            rows   [int]
    
            Outputs:
            None
    
            Properties Used:
            None
        """         
        
        # store
        self._size = rows
        
        for k,v in self.items(): 
            try:
                rank = v.ndim
            except:
                rank = 0            
            # don't expand initials or numerics
            if k in ('initials','numerics'):
                continue
            
            # recursion
            elif isinstance(v,Conditions):
                v.expand_rows(rows)
            # need arrays here
            elif rank == 2:
                self[k] = np.resize(v,[rows,v.shape[1]])
            #: if type
        #: for each key,value        
        
        
## @ingroup Analyses-Mission-Segments-Conditions        
class Container(State):
    def __defaults__(self):
        """ This sets the default values.
    
            Assumptions:
            Puts the segments in the right order
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """         
        self.segments = DataOrdered()
        
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
        
        for i,(tag,sub_state) in enumerate(self.segments.items()):
            for key in ['unknowns','conditions','residuals']:
                if i == 0:
                    state_out[key].update(sub_state[key])
                else:
                    state_out[key] = state_out[key].do_recursive(append_array,sub_state[key])
            
        return state_out
        
State.Container = Container

## @ingroup Analyses-Mission-Segments-Conditions
def append_array(A,B=None):
    """ A stacking operation used by merged to put together data structures

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        A [array]
        B [array]

        Outputs:
        array

        Properties Used:
        None
    """       
    if isinstance(A,array_type) and isinstance(B,array_type):
        return np.vstack([A,B])
    else:
        return None