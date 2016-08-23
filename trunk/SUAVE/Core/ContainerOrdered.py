# ContainerOrdered.py
#
# Created:  Jan 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald
#           Jun 2016, E. Botero


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------        

from DataOrdered import DataOrdered
from warnings    import warn

# ----------------------------------------------------------------------
#   Data Container Base Class
# ----------------------------------------------------------------------        

class ContainerOrdered(DataOrdered):
    """ SUAVE.Core.Container()
        
        a dict-type container with attribute, item and index style access
        intended to hold a attribute-accessible list of Data()
        no defaults are allowed
    
    """
        
    def __defaults__(self):
        pass
    
    def __init__(self,*args,**kwarg):
        super(ContainerOrdered,self).__init__(*args,**kwarg)
        self.__defaults__()
    
    def append(self,val):
        #val = self.check_new_val(val)
        DataOrdered.append(self,val)
        
    def extend(self,vals):
        if isinstance(vals,(list,tuple)):
            for v in val: self.append(v)
        elif isinstance(vals,dict):
            self.update(vals)
        else:
            raise Exception, 'unrecognized data type'