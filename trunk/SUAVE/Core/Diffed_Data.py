## @ingroup Core
# DiffedData.py
#
# Created:  Feb 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald
#           Jun 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from copy import deepcopy
from .Container import Container as ContainerBase
from .Data import Data
from .DataOrdered import DataOrdered
import numpy as np

# ----------------------------------------------------------------------
#  Config
# ----------------------------------------------------------------------

## @ingroup Core
class Diffed_Data(Data):
    """ This is for creating a data new class where a different copy is saved.
        This is useful for creating a new configuration of a vehicle.

        Assumptions:
        N/A

        Source:
        N/A
    """    

    
    def __defaults__(self):
        """ A stub for all classes that come later
            
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """  
        self.tag    = 'config'
        self._base  = Data()
        self._diff  = Data()
        
    def __init__(self,base=None):
        """ Initializes the new Diffed_Data() class through a deepcopy
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """  
        if base is None: base = Data()
        self._base = base
        this = deepcopy(base) # deepcopy is needed here to build configs - Feb 2016, T. MacDonald
        Data.__init__(self,this)
        
    def store_diff(self):
        """ Finds the differences and saves them
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """          
        delta = diff(self,self._base)
        self._diff = delta
        
    def pull_base(self):
        """ Updates the differences
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """          
        try: self._base.pull_base()
        except AttributeError: pass
        self.update(self._base)
        self.update(self._diff)
    
    def __str__(self,indent=''):
        """ This function is used for printing the class.
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """         
        try: 
            args = self._diff.__str__(indent)
            args += indent + '_base : ' + self._base.__repr__() + '\n'
            args += indent + '  tag : ' + self._base.tag + '\n'
            return args
        except AttributeError:     
            return Data.__str__(self,indent)
    
    def finalize(self):
        """ This just does a pull_base()
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """          
        ## dont do this here, breaks down stream dependencies
        # self.store_diff 
        
        self.pull_base()

# ----------------------------------------------------------------------
#  Config Container
# ----------------------------------------------------------------------

class Container(ContainerBase):
    """ A dict-type container with attribute, item and index style access
        intended to hold a attribute-accessible list of Data(). This is unordered.
        
        Assumptions:
        N/A
        
        Source:
        N/A
        
    """
    def append(self,value):
        """ Appends the value to the containers
        
            Assumptions:
            None
        
            Source:
            N/A
        
            Inputs:
            self
        
            Outputs:
            N/A
            
            Properties Used:
            N/A
        """         
        try: value.store_diff()
        except AttributeError: pass
        ContainerBase.append(self,value)
        
    def pull_base(self):
        """ Updates the differences
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """          
        for config in self:
            try: config.pull_base()
            except AttributeError: pass

    def store_diff(self):
        """ Finds the differences and saves them
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """          
        for config in self:
            try: config.store_diff()
            except AttributeError: pass
    
    def finalize(self):
        """ This just does a pull_base()
    
            Assumptions:
            N/A
    
            Source:
            N/A
    
            Inputs:
            N/A
    
            Outputs:
            N/A
    
            Properties Used:
            N/A    
        """        
        for config in self:
            try: config.finalize()
            except AttributeError: pass


# ------------------------------------------------------------
#  Handle Linking
# ------------------------------------------------------------

Diffed_Data.Container = Container

# ------------------------------------------------------------
#  Diffing Function
# ------------------------------------------------------------

def diff(A,B):
    """ The magic diff function that makes Diffed_Data() work

        Assumptions:
        N/A

        Source:
        N/A

        Inputs:
        A
        B

        Outputs:
        Result

        Properties Used:
        N/A    
    """      

    keys = set([])
    keys.update( A.keys() )
    keys.update( B.keys() )

    if isinstance(A,Diffed_Data):
        keys.remove('_base')
        keys.remove('_diff')

    result = type(A)()
    result.clear()

    for key in keys:
        va = A.get(key,None)
        vb = B.get(key,None)
        if isinstance(va,Data) and isinstance(vb,Data):
            sub_diff = diff(va,vb)
            if sub_diff:
                result[key] = sub_diff

        elif isinstance(va,Data) or isinstance(vb,Data):
            result[key] = va
            
        elif isinstance(va,DataOrdered) and isinstance(vb,DataOrdered):
            sub_diff = diff(va,vb)
            if sub_diff:
                result[key] = sub_diff

        elif isinstance(va,DataOrdered) or isinstance(vb,DataOrdered):
            result[key] = va        

        elif not np.all(va == vb):
            result[key] = va

    return result    