# Bunch.py
#
# Created:  Aug 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from Dict import Dict
#from Property import Property

# ----------------------------------------------------------------------
#   Bunch 
# ----------------------------------------------------------------------

class Bunch(Dict):
    """ A dictionary that provides attribute-style access.
        This implementation does not extend __getattribute__ to maintain
        performance.
        
    """
    
    __getitem__ = Dict.__getattribute__
        
    def clear(self):
        self.__dict__.clear()

    def get(self,k,d=None):
        return self.__dict__.get(k,d)
    
    def has_key(self,k):
        return self.__dict__.has_key(k)

    def to_dict(self):
        r = {}
        for k,v in self.items():
            if isinstance(v,Bunch):
                v = v.to_dict()
            r[k] = v
        return r
    
    def __len__(self):
        return self.__dict__.__len__()