## @ingroup Core
# Container.py
#
# Created:  Jan 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald
#           Jun 2016, E. Botero
#           May 2020, E. Botero


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------        

from .Data     import Data
from warnings import warn
import random

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
            This overrides the Data class append by allowing for duplicate named components
            The following components will get new names.
        
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
        
        # See if the item tag exists, if it does modify the name
        keys = self.keys()
        if str.lower(val.tag) in keys:
            string_of_keys = "".join(self.keys())
            n_comps = string_of_keys.count(val.tag)
            val.tag = val.tag + str(n_comps+1)
            
            # Check again, because theres an outside chance that its duplicate again. Then assign a random
            if str.lower(val.tag) in keys:
                val.tag = val.tag + str(n_comps+random.randint(0,1000))
        
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
        
    def get_children(self):
        """ Returns the components that can go inside
        
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs:
        None
    
        Outputs:
        None
    
        Properties Used:
        N/A
        """        
        
        return []    