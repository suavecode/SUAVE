## @ingroup Analyses-Mission-Segments-Conditions
# Conditions.py
#
# Created:  
# Modified: Feb 2016, A. Wendorff
#           Jun 2017, E. Botero
#           Jan 2020, M. Clarke

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
        return np.ones([self._size-1,cols])    
    
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
    
    
    def expand_rows(self,rows):
        """ Makes a 1-D array the right size. Often used after a mission is initialized to size out the vectors to the
            right size.
        
            Assumptions:
            None
    
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
        
        # recursively initialize condition and unknown arrays 
        # to have given row length
        
        for k,v in self.items():
            try:
                rank = v.ndim
            except:
                rank = 0
            # recursion
            if isinstance(v,Conditions):
                v.expand_rows(rows)
            # need arrays here
            elif rank == 2:
                self[k] = np.resize(v,[rows,v.shape[1]])
            #: if type
        #: for each key,value
        
        return

    def compile(self):
        """ This is a call to expand_rows above...
        
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
        self.expand_rows()