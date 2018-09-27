## @ingroup Core
# Container.py
#
# Created:  Jan 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald
#           Jun 2016, E. Botero

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------        

from .Data     import Data
from warnings import warn

# ----------------------------------------------------------------------
#   Data Container Base Class
# ----------------------------------------------------------------------        

## @ingroup Core
class Container(Data):
    """ A dict-type container with attribute, item and index style access
        intended to hold a attribute-accessible list of Data(). This is unordered.
        
        Assumptions:
        N/A
        
        Source:
        N/A
        
    """
            
        
    def __defaults__(self):
        """ Defaults function
    
            Assumptions:
            None
        
            Source:
            N/A
        
            Inputs:
            N/A
        
            Outputs:
            N/A
            
            Properties Used:
            N/A
        """          
        pass
    
    def __init__(self,*args,**kwarg):
        """ Initialization that builds the container
        
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
        super(Container,self).__init__(*args,**kwarg)
        self.__defaults__()
    
    def append(self,val):
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
        
        #val = self.check_new_val(val)
        Data.append(self,val)
        
    def extend(self,vals):
        """ Append things regressively depending on what is inside.
    
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
        if isinstance(vals,(list,tuple)):
            for v in val: self.append(v)
        elif isinstance(vals,dict):
            self.update(vals)
        else:
            raise Exception('unrecognized data type')