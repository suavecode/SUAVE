## @ingroup Analyses-Mission-Segments-Conditions
# Conditions.py
#
# Created:  
# Modified: Feb 2016, A. Wendorff
#           Jun 2017, E. Botero
#           Jan 2020, M. Clarke
#           Oct 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# python imports
import numpy as np

# SUAVE imports
from SUAVE.Core                    import Data

# ----------------------------------------------------------------------
#  Conditions
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Conditions
class Conditions(Data):
    """ Conditions are the magic Data that contains the information about the vehicle in flight.
        At this point none of the information really exists. What is here are the methods that allow a mission
        to collect the information.
    
        Assumptions:
        None
        
        Source:
        None   
    """ 

    _size = 1
    
    def ones_row(self,cols):
        """ returns a row vector of ones with given number of columns 
        
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            cols   [in]
    
            Outputs:
            Vector
    
            Properties Used:
            None
        """     
        return np.ones([self._size,cols])
    
    def ones_row_m1(self,cols):
        """ returns an N-1 row vector of ones with given number of columns
        
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            cols   [in]
    
            Outputs:
            Vector
    
            Properties Used:
            None
        """ 
        #return np.ones([self._size-1,cols])  
        return expanded_array(cols, 1)
    
    def ones_row_m2(self,cols):
        """ returns an N-2 row vector of ones with given number of columns
        
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            cols   [int]
    
            Outputs:
            Vector
    
            Properties Used:
            None
        """ 
        return np.ones([self._size-2,cols])
    
    
    def expand_rows(self,rows,override=False):
        """ Makes a 1-D array the right size. Often used after a mission is initialized to size out the vectors to the
            right size. Will not overwrite an array if it already exists, unless override is True.
        
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            rows     [int]
            override [boolean]
    
            Outputs:
            None
    
            Properties Used:
            None
        """           
        
        # store
        self._size = rows
        
        # recursively initialize condition and unknown arrays 
        # to have given row length
        
        for k,v in self.items():
            try:
                rank = v.ndim
            except:
                rank = 0
            # recursion
            if isinstance(v,Conditions):
                v.expand_rows(rows,override=override)
            elif isinstance(v,expanded_array):
                self[k] = v.resize(rows, v)
            # need arrays here
            elif rank == 2:
                #Check if it's already expanded
                if v.shape[0]<=1 or override:
                    self[k] = np.resize(v,[rows,v.shape[1]])
        
        return
        
## @ingroup Analyses-Mission-Segments-Conditions        
class expanded_array(Data):
    """"""

    _size = 1  
        
    def __init__(self, cols, adjustment):
        """ Initialization that sets expansion later
        
            Assumptions:
            None
        
            Source:
            N/A
        
            Inputs:
            self
            size       - usually number of control points [int]
            cols       - columns                          [int]
            adjustment - how much smaller                 [int]
        
            Outputs:
            N/A
            
            Properties Used:
            N/A
        """          
        
        self._adjustment = adjustment
        self._cols       = cols
        
        
    def resize(self,rows,v):
        """"""
        # unpack
        adjustment = self._adjustment
        
        # pack
        self._size = rows
        value      = v()
        
        return np.resize(value,[rows-adjustment,value.shape[1]])
    
    def __call__(self):
        
        return self._array
    
    def __mul__(self,other):

        
        self._array = np.resize(other.value,[1,1])
        
        return self

    def __rmul__(self,other):

        
        self._array = np.resize(other,[1,1])
        
        return self    
        
    