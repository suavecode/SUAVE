# Container.py
#
# Created:  Jan 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald


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
            raise Data_Exception, 'unrecognized data type'
        
    def check_new_val(self,val):
        
        # make sure val is a Data()
        if not isinstance(val,Data): 
            raise Data_Exception , 'val must be a Data() instance'        
        
        # make sure val has a tag
        if not val.has_key('tag'): 
            raise Data_Exception , 'val.tag must exist and be unique'
        
        # make sure tag is unique
        ns = len(val.tag)
        if self.has_key(val.tag):
            #raise Data_Exception , 'val.tag=%s must exist and be unique'%val.tag
            warn('\nval.tag should be unique',Data_Warning)
            # add index to new val
            keys = [k for k in self.iterkeys() if k[:(ns)] == val.tag]
            val.tag = val.tag + '_%i' % (len(keys)+1)
            # add index to existing val
            if len(keys) == 1:
                key_old = keys[0]
                key_new = key_old + '_1'
                self[key_old].tag = key_new
                self[key_new] = self[key_old]
                del self[key_old]
            
        return val
        
