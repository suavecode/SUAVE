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
        return expanded_array(cols, 2)
    
    
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
                self[k] = v.resize(rows)
            # need arrays here
            elif rank == 2:
                #Check if it's already expanded
                if v.shape[0]<=1 or override:
                    self[k] = np.resize(v,[rows,v.shape[1]])
        
        return
        
## @ingroup Analyses-Mission-Segments-Conditions        
class expanded_array(Data):
    """ This is an array that will expand later when the mission is initialized. It is called specifically by conditions
    
        Assumptions:
        None
        
        Source:
        None   
    """ 

    _size = 1  
        
    def __init__(self, cols, adjustment):
        """ Initialization that sets expansion later
        
            Assumptions:
            None
        
            Source:
            N/A
        
            Inputs:
            self
            cols       - columns                          [int]
            adjustment - how much smaller                 [int]
        
            Outputs:
            N/A
            
            Properties Used:
            N/A
        """          
        
        self._adjustment = adjustment
        self._cols       = cols
        self._array      = np.array([[1]])
        
        
    def resize(self,rows):
        """ This function actually completes the resizing. After this it's no longer an expanded array. That way it
            doesn't propogate virally. That means if one wishes to resize later the conditions need to be reset.
        
            Assumptions:
            None
        
            Source:
            N/A
        
            Inputs:
            self
            rows       - rows                             [int]
            v          - values (really self)             [int]
        
            Outputs:
            np.array   - properly sized                   [array]
            
            Properties Used:
            N/A
        """   
        # unpack
        adjustment = self._adjustment
        
        # pack
        self._size = rows
        value      = self._array
        
        return np.resize(value,[rows-adjustment,value.shape[1]])
    
    def __call__(self):
        """ This returns the value and shape of the array as is
        
            Assumptions:
            None
        
            Source:
            N/A
        
            Inputs:
            self

            Outputs:
            np.array   - properly sized                   [array]
            
            Properties Used:
            N/A
        """           
        
        return self._array
    
    def __mul__(self,other):
        """ Performs multiplication and returns self
        
            Assumptions:
            None
        
            Source:
            N/A
        
            Inputs:
            self
            other      - something can be multiplied      [float]

            Outputs:
            self
            
            Properties Used:
            N/A
        """          
        
        self._array = np.resize(other,[1,1])
        
        return self

    def __rmul__(self,other):
        """ Performs multiplication and returns self
        
            Assumptions:
            None
        
            Source:
            N/A
        
            Inputs:
            self
            other      - something can be multiplied      [float]

            Outputs:
            self
            
            Properties Used:
            N/A
        """                 
        
        self._array = np.resize(other,[1,1])
        
        return self    
        
    