
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np
from copy import deepcopy

# SUAVE imports
from SUAVE.Core                    import Data, Data_Exception
from SUAVE.Methods.Utilities            import atleast_2d_col


# ----------------------------------------------------------------------
#  Conditions
# ----------------------------------------------------------------------

class Conditions(Data):

    _size = 1
    
    def ones_row(self,cols):
        """ returns a row vector of ones with given number of columns """
        return np.ones([self._size,cols])
    
    def expand_rows(self,rows):
        
        # store
        self._size = rows
        
        # recursively initialize condition and unknown arrays 
        # to have given row length
        
        for k,v in self.iteritems():
            # recursion
            if isinstance(v,Conditions):
                v.expand_rows(rows)
            # need arrays here
            elif np.rank(v) == 2:
                self[k] = np.resize(v,[rows,v.shape[1]])
            #: if type
        #: for each key,value
        
        return

    def compile(self):
        self.expand_rows()