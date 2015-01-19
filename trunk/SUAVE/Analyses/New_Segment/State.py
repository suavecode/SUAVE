
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np
from copy import deepcopy

# SUAVE imports
from SUAVE.Core                    import Data, Data_Exception
from SUAVE.Methods.Utilities            import atleast_2d_col

from Conditions import Conditions, Unknowns, Residuals, Numerics


# ----------------------------------------------------------------------
#  State
# ----------------------------------------------------------------------

class State(Conditions):
    
    def __defaults__(self):
        
        self.unknowns   = Unknowns()
        
        self.conditions = Conditions()
        
        self.residuals  = Residuals()
        
        self.numerics   = Numerics()
        
        self.initials   = Conditions()
        
        
    def expand_rows(self,rows):
        
        # store
        self._size = rows
        
        for k,v in self.iteritems():
            
            # don't expand initials or numerics
            if k in ('initials','numerics'):
                continue
            
            # recursion
            elif isinstance(v,Conditions):
                v.expand_rows(rows)
            # need arrays here
            elif np.rank(v) == 2:
                self[k] = np.resize(v,[rows,v.shape[1]])
            #: if type
        #: for each key,value        
        